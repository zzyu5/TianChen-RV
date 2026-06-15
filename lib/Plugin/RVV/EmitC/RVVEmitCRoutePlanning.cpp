#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "RVVEmitCRoutePlanningInternal.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"
#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine(
          "TianChen-RV RVV plugin-owned EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeRVVSelectedTargetCapabilityError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV selected target-capability gate failed: ") +
          message,
      llvm::errc::invalid_argument);
}

constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kProfileCapabilityKind("profile");
constexpr llvm::StringLiteral kArchitecturePropertyName("architecture");
constexpr llvm::StringLiteral kISAVectorHintsPropertyName("isa_vector_hints");
constexpr llvm::StringLiteral kSupportedSEWPropertyName("supported_sew");
constexpr llvm::StringLiteral kSupportedLMULPropertyName("supported_lmul");
constexpr llvm::StringLiteral kRequiredTailPolicyPropertyName(
    "required_tail_policy");
constexpr llvm::StringLiteral kRequiredMaskPolicyPropertyName(
    "required_mask_policy");
std::string joinRVVSelectedCapabilityProviderSymbols(
    llvm::ArrayRef<const support::CapabilityDescriptor *> providers) {
  std::string symbols;
  llvm::raw_string_ostream stream(symbols);
  for (std::size_t index = 0, count = providers.size(); index < count;
       ++index) {
    if (index != 0)
      stream << ", ";
    stream << "@" << providers[index]->getSymbolName();
  }
  return stream.str();
}

llvm::StringRef classifyRVVSatisfaction(
    const support::CapabilityDescriptor &capability) {
  if (capability.getID() == kRVVCapabilityID)
    return "exact";
  if (capability.providesID(kRVVCapabilityID))
    return "provides";
  if (capability.impliesID(kRVVCapabilityID))
    return "implies";
  return "unknown";
}

bool capabilityPropertyListContains(llvm::StringRef value,
                                    llvm::StringRef expectedToken) {
  llvm::SmallVector<llvm::StringRef, 8> tokens;
  llvm::SplitString(value, tokens, ",;| \t\r\n");
  for (llvm::StringRef token : tokens)
    if (token.trim() == expectedToken)
      return true;
  return false;
}

bool containsRVVVectorISAHint(llvm::StringRef hints) {
  std::string lower = hints.lower();
  llvm::StringRef text(lower);
  return text.contains("rv64gcv") || text.contains("rv32gcv") ||
         text.contains("zve") || text.contains("zvl") ||
         text.contains("zvfh") || text.contains("rvv");
}

std::string formatRVVSelectedCapabilityProviderMirror(
    const support::CapabilityDescriptor &capability) {
  std::string mirror;
  llvm::raw_string_ostream stream(mirror);
  stream << "selected_capability_provider_mirror:@"
         << capability.getSymbolName() << ";id=" << capability.getID()
         << ";kind=" << capability.getKind()
         << ";rvv=" << classifyRVVSatisfaction(capability);
  return stream.str();
}

std::string formatRVVSelectedCapabilityLegalityMirror(
    const RVVSelectedTargetCapabilityFacts &facts,
    const RVVSelectedBodyTypedConfigFacts &typedConfigFacts) {
  std::string mirror;
  llvm::raw_string_ostream stream(mirror);
  stream << "selected_target_capability_legality_mirror:@"
         << facts.selectedProviderSymbol << ";id=" << facts.selectedProviderID
         << ";kind=" << facts.selectedProviderKind
         << ";rvv=" << facts.rvvSatisfactionKind
         << ";sew=" << typedConfigFacts.sew
         << ";lmul=" << typedConfigFacts.lmul
         << ";tail=" << typedConfigFacts.tailPolicy
         << ";mask=" << typedConfigFacts.maskPolicy;
  return stream.str();
}

llvm::Error verifyRVVCapabilityProfileProperties(
    const support::CapabilityDescriptor &capability, llvm::StringRef context) {
  llvm::StringRef architecture =
      capability.getProperty(kArchitecturePropertyName).trim();
  if (!architecture.empty() && architecture != "riscv64")
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " architecture fact '" + architecture +
        "' is incompatible with RVV selected routes; expected 'riscv64'");

  llvm::StringRef isaVectorHints =
      capability.getProperty(kISAVectorHintsPropertyName).trim();
  if (!isaVectorHints.empty() && !containsRVVVectorISAHint(isaVectorHints))
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " isa_vector_hints fact '" +
        isaVectorHints + "' does not contain RVV vector ISA evidence");

  return llvm::Error::success();
}

llvm::Expected<RVVSelectedTargetCapabilityFacts>
collectRVVSelectedTargetCapabilityFacts(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    llvm::StringRef context) {
  if (!variant)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) +
        " requires a materialized tcrv.exec.variant");

  auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>(
      kRequiresAttrName);
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
          " does not resolve in the tcrv.exec target capability set");

    if (capability->satisfiesID(kRVVCapabilityID))
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
        joinRVVSelectedCapabilityProviderSymbols(selectedRVVProviders));

  const support::CapabilityDescriptor &capability =
      *selectedRVVProviders.front();
  if (!capability.isAvailable())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " satisfying id 'rvv' is unavailable "
        "(status = '" +
        capability.getStatus() + "')");

  if (capability.getID() == kRVVCapabilityID &&
      capability.getKind() != kRVVCapabilityKind)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() + " declares exact id 'rvv' but kind '" +
        capability.getKind() + "'; expected kind 'isa-vector'");
  if (capability.getID() != kRVVCapabilityID &&
      capability.getKind() != kRVVCapabilityKind &&
      capability.getKind() != kProfileCapabilityKind)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        capability.getSymbolName() +
        " satisfying id 'rvv' must be kind 'isa-vector' or 'profile', got '" +
        capability.getKind() + "'");

  if (llvm::Error error =
          verifyRVVCapabilityProfileProperties(capability, context))
    return std::move(error);

  RVVSelectedTargetCapabilityFacts facts;
  facts.selectedProviderSymbol = capability.getSymbolName().str();
  facts.selectedProviderID = capability.getID().str();
  facts.selectedProviderKind = capability.getKind().str();
  facts.rvvSatisfactionKind = classifyRVVSatisfaction(capability).str();
  facts.supportedSEW =
      capability.getProperty(kSupportedSEWPropertyName).trim().str();
  facts.supportedLMUL =
      capability.getProperty(kSupportedLMULPropertyName).trim().str();
  facts.requiredTailPolicy =
      capability.getProperty(kRequiredTailPolicyPropertyName).trim().str();
  facts.requiredMaskPolicy =
      capability.getProperty(kRequiredMaskPolicyPropertyName).trim().str();
  facts.providerMirror = formatRVVSelectedCapabilityProviderMirror(capability);
  return facts;
}

llvm::Error verifyRVVSelectedTargetCapabilityForTypedConfig(
    RVVSelectedTargetCapabilityFacts &facts,
    const RVVSelectedBodyTypedConfigFacts &typedConfigFacts,
    llvm::StringRef context) {
  if (!facts.hasFacts())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) +
        " requires collected selected RVV target capability facts");
  if (!typedConfigFacts.hasFacts())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) +
        " requires typed RVV body/config facts before capability gating");

  std::string expectedSEW = llvm::Twine(typedConfigFacts.sew).str();
  if (!facts.supportedSEW.empty() &&
      !capabilityPropertyListContains(facts.supportedSEW, expectedSEW))
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " supported_sew fact '" +
        facts.supportedSEW + "' does not include typed body SEW " +
        expectedSEW);

  if (!facts.supportedLMUL.empty() &&
      !capabilityPropertyListContains(facts.supportedLMUL,
                                      typedConfigFacts.lmul))
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " supported_lmul fact '" +
        facts.supportedLMUL + "' does not include typed body LMUL '" +
        typedConfigFacts.lmul + "'");

  if (!facts.requiredTailPolicy.empty() &&
      llvm::StringRef(facts.requiredTailPolicy) != typedConfigFacts.tailPolicy)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " required_tail_policy fact '" +
        facts.requiredTailPolicy + "' does not match typed body tail policy '" +
        typedConfigFacts.tailPolicy + "'");

  if (!facts.requiredMaskPolicy.empty() &&
      llvm::StringRef(facts.requiredMaskPolicy) != typedConfigFacts.maskPolicy)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " required_mask_policy fact '" +
        facts.requiredMaskPolicy + "' does not match typed body mask policy '" +
        typedConfigFacts.maskPolicy + "'");

  facts.legalityMirror =
      formatRVVSelectedCapabilityLegalityMirror(facts, typedConfigFacts);
  return llvm::Error::success();
}

struct RVVSelectedDispatchEnvelopeFacts {
  std::string selectedDispatchCaseMirror;
  std::string selectedDispatchFallbackMirror;
  RVVLowPrecisionSelectedDispatchPolicyBoundary
      lowPrecisionSelectedDispatchPolicyBoundary;

  bool hasFacts() const { return !selectedDispatchCaseMirror.empty(); }
};

constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
constexpr llvm::StringLiteral kRuntimeGuardRequiredAttrName(
    "runtime_guard_required");
constexpr llvm::StringLiteral kRuntimeGuardAttrName("runtime_guard");
constexpr llvm::StringLiteral kABIRoleAttrName("abi_role");
constexpr llvm::StringLiteral kDispatchAvailabilityGuardRoleValue(
    "dispatch-availability-guard");

tcrv::exec::VariantOp findDirectKernelVariant(tcrv::exec::KernelOp kernel,
                                              llvm::StringRef symbol) {
  if (!kernel || kernel.getBody().empty())
    return {};

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
    if (variant && variant.getSymName() == symbol)
      return variant;
  }
  return {};
}

mlir::Operation *findDirectKernelSymbol(tcrv::exec::KernelOp kernel,
                                        llvm::StringRef symbol) {
  if (!kernel || kernel.getBody().empty())
    return nullptr;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto symbolAttr = op.getAttrOfType<mlir::StringAttr>("sym_name");
    if (symbolAttr && symbolAttr.getValue() == symbol)
      return &op;
  }
  return nullptr;
}

std::string getOptionalStringAttr(mlir::Operation *op,
                                  llvm::StringRef attrName) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  return attr ? attr.getValue().str() : std::string();
}

llvm::Error verifyDispatchRuntimeGuardLink(tcrv::exec::KernelOp kernel,
                                           tcrv::exec::DispatchCaseOp op,
                                           llvm::StringRef context,
                                           std::string &guardMirror) {
  auto required = op->getAttrOfType<mlir::BoolAttr>(
      kRuntimeGuardRequiredAttrName);
  if (!required || !required.getValue()) {
    guardMirror = "runtime_guard_required=false;runtime_guard=none";
    return llvm::Error::success();
  }

  auto guard = op->getAttrOfType<mlir::FlatSymbolRefAttr>(
      kRuntimeGuardAttrName);
  if (!guard)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " selected dispatch case declares runtime_guard_required=true but "
        "does not link runtime_guard to a same-kernel "
        "dispatch-availability-guard runtime_param before RVV route "
        "construction");

  mlir::Operation *resolved = findDirectKernelSymbol(kernel, guard.getValue());
  if (!resolved)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " selected dispatch case runtime_guard @" +
        guard.getValue() + " does not resolve in the same tcrv.exec.kernel");

  auto runtimeParam = llvm::dyn_cast<tcrv::exec::RuntimeParamOp>(resolved);
  if (!runtimeParam)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " selected dispatch case runtime_guard @" +
        guard.getValue() +
        " resolves to a direct sibling symbol that is not a "
        "tcrv.exec.runtime_param");

  auto role = runtimeParam->getAttrOfType<mlir::StringAttr>(kABIRoleAttrName);
  if (!role || role.getValue() != kDispatchAvailabilityGuardRoleValue)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " selected dispatch case runtime_guard @" +
        guard.getValue() +
        " must reference a tcrv.exec.runtime_param with ABI role '" +
        kDispatchAvailabilityGuardRoleValue + "'");

  guardMirror =
      (llvm::Twine("runtime_guard_required=true;runtime_guard=@") +
       guard.getValue())
          .str();
  return llvm::Error::success();
}

std::string formatRVVSelectedDispatchCaseMirror(
    tcrv::exec::DispatchCaseOp dispatchCase, tcrv::exec::VariantOp variant,
    llvm::StringRef guardMirror) {
  std::string mirror;
  llvm::raw_string_ostream stream(mirror);
  stream << "selected_dispatch_case_mirror:@" << variant.getSymName()
         << ";role=" << stringifyVariantEmissionRole(
                            VariantEmissionRole::DispatchCase)
         << ";" << guardMirror;
  std::string origin = getOptionalStringAttr(dispatchCase.getOperation(),
                                             kOriginAttrName);
  if (!origin.empty())
    stream << ";origin=" << origin;
  std::string policy = getOptionalStringAttr(dispatchCase.getOperation(),
                                             kPolicyAttrName);
  if (!policy.empty())
    stream << ";policy=" << policy;
  return stream.str();
}

std::string formatRVVSelectedDispatchFallbackMirror(
    tcrv::exec::FallbackOp fallback, tcrv::exec::VariantOp variant) {
  std::string mirror;
  llvm::raw_string_ostream stream(mirror);
  stream << "selected_dispatch_fallback_mirror:@" << variant.getSymName()
         << ";role=" << stringifyVariantEmissionRole(
                            VariantEmissionRole::DispatchFallback)
         << ";fallback_role=" << kConservativeFallbackRoleValue;
  std::string origin = getOptionalStringAttr(fallback.getOperation(),
                                             kOriginAttrName);
  if (!origin.empty())
    stream << ";origin=" << origin;
  std::string policy =
      getOptionalStringAttr(fallback.getOperation(), kPolicyAttrName);
  if (!policy.empty())
    stream << ";policy=" << policy;
  return stream.str();
}

llvm::Expected<RVVSelectedDispatchEnvelopeFacts>
collectRVVSelectedDispatchEnvelopeFacts(
    const VariantEmitCLowerableRequest &request, llvm::StringRef context) {
  RVVSelectedDispatchEnvelopeFacts facts;
  if (request.getRole() == VariantEmissionRole::DirectVariant)
    return facts;

  if (request.getRole() == VariantEmissionRole::DispatchFallback)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " refuses RVV route construction for a dispatch fallback role; "
        "fallback linkage is an explicit tcrv.exec envelope boundary and is "
        "not RVV route authority");

  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp selectedVariant = request.getVariant();
  if (!kernel || kernel.getBody().empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a materialized tcrv.exec.kernel with a selected dispatch");
  if (!selectedVariant)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a materialized selected tcrv.exec.variant");

  tcrv::exec::DispatchOp selectedDispatch;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto dispatch = llvm::dyn_cast<tcrv::exec::DispatchOp>(op);
    if (!dispatch)
      continue;
    if (selectedDispatch)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " requires exactly one direct tcrv.exec.dispatch before RVV "
          "dispatch-case route construction");
    selectedDispatch = dispatch;
  }
  if (!selectedDispatch)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires a direct tcrv.exec.dispatch before RVV dispatch-case "
        "route construction");

  tcrv::exec::DispatchCaseOp selectedCase;
  tcrv::exec::FallbackOp fallbackOp;
  tcrv::exec::VariantOp fallbackVariant;
  unsigned caseCount = 0;
  unsigned fallbackCount = 0;

  for (mlir::Operation &op : selectedDispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<tcrv::exec::DispatchCaseOp>(op)) {
      ++caseCount;
      auto target =
          dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
              kTargetAttrName);
      if (target && target.getValue() == selectedVariant.getSymName()) {
        if (selectedCase)
          return makeRVVEmitCRouteProviderError(
              llvm::Twine(context) +
              " found multiple dispatch cases for the selected RVV variant @"
              + selectedVariant.getSymName());
        selectedCase = dispatchCase;
      }
      continue;
    }

    if (auto fallback = llvm::dyn_cast<tcrv::exec::FallbackOp>(op)) {
      ++fallbackCount;
      auto target =
          fallback->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
      if (!target)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " selected dispatch fallback is missing a variant target");
      fallbackVariant = findDirectKernelVariant(kernel, target.getValue());
      if (!fallbackVariant)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) + " selected dispatch fallback target @" +
            target.getValue() +
            " does not resolve to a direct sibling tcrv.exec.variant");
      fallbackOp = fallback;
      continue;
    }
  }

  if (caseCount == 0 || !selectedCase)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires the selected RVV variant to be referenced by a "
        "tcrv.exec.dispatch case before route construction");
  if (fallbackCount != 1 || !fallbackOp || !fallbackVariant)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires exactly one explicit tcrv.exec.fallback target before RVV "
        "dispatch-case route construction");

  auto fallbackRole = fallbackVariant->getAttrOfType<mlir::StringAttr>(
      kFallbackRoleAttrName);
  if (!fallbackRole ||
      fallbackRole.getValue() != kConservativeFallbackRoleValue)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " selected dispatch fallback target @" +
        fallbackVariant.getSymName() +
        " must be a fallback-eligible tcrv.exec.variant with "
        "fallback_role='conservative'");

  std::string guardMirror;
  if (llvm::Error error =
          verifyDispatchRuntimeGuardLink(kernel, selectedCase, context,
                                         guardMirror))
    return std::move(error);

  facts.selectedDispatchCaseMirror =
      formatRVVSelectedDispatchCaseMirror(selectedCase, selectedVariant,
                                          guardMirror);
  facts.selectedDispatchFallbackMirror =
      formatRVVSelectedDispatchFallbackMirror(fallbackOp, fallbackVariant);
  RVVLowPrecisionSelectedDispatchPolicyBoundary &policyBoundary =
      facts.lowPrecisionSelectedDispatchPolicyBoundary;
  policyBoundary.hasSelectedDispatchCase = true;
  policyBoundary.hasSelectedDispatchFallback = true;
  policyBoundary.selectedCaseVariant = selectedVariant.getSymName().str();
  policyBoundary.selectedCaseRole =
      stringifyVariantEmissionRole(VariantEmissionRole::DispatchCase).str();
  policyBoundary.selectedCaseOrigin =
      getOptionalStringAttr(selectedCase.getOperation(), kOriginAttrName);
  policyBoundary.selectedCasePolicy =
      getOptionalStringAttr(selectedCase.getOperation(), kPolicyAttrName);
  auto runtimeGuardRequired = selectedCase->getAttrOfType<mlir::BoolAttr>(
      kRuntimeGuardRequiredAttrName);
  policyBoundary.runtimeGuardRequired =
      runtimeGuardRequired && runtimeGuardRequired.getValue();
  if (policyBoundary.runtimeGuardRequired) {
    auto runtimeGuard = selectedCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kRuntimeGuardAttrName);
    policyBoundary.runtimeGuard =
        runtimeGuard ? (llvm::Twine("@") + runtimeGuard.getValue()).str()
                     : std::string();
  } else {
    policyBoundary.runtimeGuard = "none";
  }
  policyBoundary.fallbackVariant = fallbackVariant.getSymName().str();
  policyBoundary.fallbackPathRole =
      stringifyVariantEmissionRole(VariantEmissionRole::DispatchFallback)
          .str();
  policyBoundary.fallbackRole = kConservativeFallbackRoleValue.str();
  policyBoundary.fallbackOrigin =
      getOptionalStringAttr(fallbackOp.getOperation(), kOriginAttrName);
  policyBoundary.fallbackPolicy =
      getOptionalStringAttr(fallbackOp.getOperation(), kPolicyAttrName);
  policyBoundary.selectedDispatchCaseMirror =
      facts.selectedDispatchCaseMirror;
  policyBoundary.selectedDispatchFallbackMirror =
      facts.selectedDispatchFallbackMirror;
  return facts;
}

constexpr llvm::StringLiteral
    kRVVRuntimeScalarCompareSelectOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_select.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarDualCompareMaskAndSelectOperandBindingPlanID(
        "rvv-route-operand-binding:rs_dual_cmp_mask_select.v1");
constexpr llvm::StringLiteral kRVVF32ClampSelectOperandBindingPlanID(
    "rvv-route-operand-binding:f32_clamp_select.v1");
constexpr llvm::StringLiteral
    kRVVDequantClampF32EpilogueOperandBindingPlanID(
        "rvv-route-operand-binding:dequant_clamp_f32_epilogue.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionOperandBindingPlanID(
    "rvv-route-operand-binding:standalone_reduce_add.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionMinOperandBindingPlanID(
    "rvv-route-operand-binding:standalone_reduce_min.v1");
constexpr llvm::StringLiteral kRVVStandaloneReductionMaxOperandBindingPlanID(
    "rvv-route-operand-binding:standalone_reduce_max.v1");
constexpr llvm::StringLiteral
    kRVVWideningStandaloneReductionOperandBindingPlanID(
        "rvv-route-operand-binding:widening_standalone_reduce_add.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionOperandBindingPlanID(
        "rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionMinOperandBindingPlanID(
        "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskStandaloneReductionMaxOperandBindingPlanID(
        "rvv-route-operand-binding:computed_mask_standalone_reduce_max.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionMinOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_min.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskStandaloneReductionMaxOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_max.v1");
constexpr llvm::StringLiteral kRVVCmpSelectOperandBindingPlanID(
    "rvv-route-operand-binding:cmp_select.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectOperandBindingPlanID(
    "rvv-route-operand-binding:computed_mask_select.v1");

std::optional<llvm::StringRef>
getRVVStandaloneReductionRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
    return kRVVStandaloneReductionOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
    return kRVVStandaloneReductionMinOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
    return kRVVStandaloneReductionMaxOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
    return kRVVWideningStandaloneReductionOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
    return kRVVComputedMaskStandaloneReductionOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
    return kRVVComputedMaskStandaloneReductionMinOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return kRVVComputedMaskStandaloneReductionMaxOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd:
    return kRVVRuntimeScalarComputedMaskStandaloneReductionOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMin:
    return kRVVRuntimeScalarComputedMaskStandaloneReductionMinOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMax:
    return kRVVRuntimeScalarComputedMaskStandaloneReductionMaxOperandBindingPlanID;
  default:
    return std::nullopt;
  }
}

std::optional<llvm::StringRef>
getRVVRuntimeScalarComputedMaskStandaloneReductionRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMax:
    return getRVVStandaloneReductionRouteOperandBindingPlanID(operation);
  default:
    return std::nullopt;
  }
}

bool isRVVPlainStandaloneReductionRouteFactsOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVVectorComputedMaskStandaloneReductionRouteFactsOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFactsOperation(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

llvm::StringRef
getRVVStandaloneReductionKindMirror(RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
    return "add";
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
    return "min";
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return "max";
  case RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
    return "signed_widening_reduce_add";
  default:
    return "";
  }
}

bool populateRVVStandaloneReductionRouteFacts(
    RVVStandaloneReductionRouteFacts &facts,
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    bool includeRouteOperandBindingSummary);

std::optional<RVVStandaloneReductionRouteFacts>
getRVVStandaloneReductionRouteFacts(RVVSelectedBodyOperationKind operation) {
  RVVStandaloneReductionRouteFacts facts;
  if (!populateRVVStandaloneReductionRouteFacts(
          facts, operation, /*sew=*/0,
          /*includeRouteOperandBindingSummary=*/true))
    return std::nullopt;
  return facts;
}

std::optional<RVVStandaloneReductionRouteFacts>
getRVVStandaloneReductionRouteFacts(RVVSelectedBodyOperationKind operation,
                                    std::int64_t sew) {
  RVVStandaloneReductionRouteFacts facts;
  if (!populateRVVStandaloneReductionRouteFacts(
          facts, operation, sew, /*includeRouteOperandBindingSummary=*/true))
    return std::nullopt;
  return facts;
}

bool populateRVVStandaloneReductionRouteFacts(
    RVVStandaloneReductionRouteFacts &facts,
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    bool includeRouteOperandBindingSummary) {
  std::optional<llvm::StringRef> planID =
      getRVVStandaloneReductionRouteOperandBindingPlanID(operation);
  if (!planID)
    return false;
  const bool isWideningStandalone =
      operation == RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd;
  if (isWideningStandalone && sew != 0 && sew != 32)
    return false;
  if (!isWideningStandalone && sew != 0 && sew != 32 && sew != 64)
    return false;

  const bool isPlain =
      isRVVPlainStandaloneReductionRouteFactsOperation(operation);
  const bool isVectorComputedMask =
      isRVVVectorComputedMaskStandaloneReductionRouteFactsOperation(operation);
  const bool isRuntimeScalarComputedMask =
      isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFactsOperation(
          operation);
  const bool isComputedMask = isVectorComputedMask || isRuntimeScalarComputedMask;
  if (!isPlain && !isComputedMask)
    return false;

  const bool isAdd =
      operation == RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd ||
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskStandaloneReduceAdd;
  const llvm::StringRef inactiveUse =
      isComputedMask ? (isAdd ? llvm::StringRef("zero-inactive")
                              : llvm::StringRef("neutral-inactive"))
                     : llvm::StringRef();
  const llvm::StringRef inactiveRequirement =
      isComputedMask
          ? (isAdd
                 ? llvm::StringRef(
                       "masked-standalone-reduction-zero-inactive-lanes-before-reduction")
                 : llvm::StringRef(
                       "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"))
          : llvm::StringRef();

  facts.operation = operation;
  facts.reductionKind = getRVVStandaloneReductionKindMirror(operation);
  facts.memoryForm =
      isRuntimeScalarComputedMask
          ? RVVSelectedBodyMemoryForm::
                RuntimeScalarComputedMaskUnitStrideStandaloneReduction
      : isVectorComputedMask
          ? RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideStandaloneReduction
          : RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction;
  facts.typedComputeOpName =
      isComputedMask ? llvm::StringRef("tcrv_rvv.masked_standalone_reduce")
                     : llvm::StringRef("tcrv_rvv.standalone_reduce");
  facts.runtimeABIOrder =
      isRuntimeScalarComputedMask
          ? llvm::StringRef("cmp_lhs,rhs_scalar,src,acc,out,n")
      : isVectorComputedMask ? llvm::StringRef("cmp_lhs,cmp_rhs,src,acc,out,n")
                             : llvm::StringRef("lhs,acc,out,n");
  facts.targetLeafProfile =
      isWideningStandalone
          ? llvm::StringRef(
                "rvv-v1-typed-i16-to-i32-widening-standalone-reduction-leaf-profile.v1")
      : isRuntimeScalarComputedMask
          ? llvm::StringRef(
                "rvv-v1-typed-runtime-scalar-cmp-masked-standalone-reduction-leaf-profile.v1")
      : isVectorComputedMask
          ? llvm::StringRef(
                "rvv-v1-typed-computed-mask-standalone-reduction-leaf-profile.v1")
          : llvm::StringRef("rvv-v1-typed-standalone-reduction-leaf-profile.v1");
  facts.providerSupportedMirror =
      isWideningStandalone
          ? llvm::StringRef(
                "provider_supported_mirror:rvv-i16-to-i32-widening-standalone-reduction-plan-validated")
      : isRuntimeScalarComputedMask
          ? llvm::StringRef(
                "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated")
      : isVectorComputedMask
          ? llvm::StringRef(
                "provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated")
          : llvm::StringRef(
                "provider_supported_mirror:rvv-standalone-reduction-plan-validated");
  facts.requiredHeaderDeclarations = "stddef.h,stdint.h,riscv_vector.h";
  facts.cTypeMappingSummary =
      isWideningStandalone
          ? llvm::StringRef(
                "vl:size_t,input:signed-e16mf2,seed:signed-i32,result:signed-e32m1")
      : isRuntimeScalarComputedMask
          ? llvm::StringRef(
                "vl:size_t,cmp_lhs/source:typed-source-vector,rhs_scalar:typed-scalar,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector")
      : isVectorComputedMask
          ? llvm::StringRef(
                "vl:size_t,compare/source:typed-source-vector,mask:typed-mask,seed:typed-scalar,result:typed-scalar-reduction-vector")
          : llvm::StringRef(
                "vl:size_t,input:typed-source-vector,seed:typed-scalar,result:typed-scalar-reduction-vector");
  facts.routeOperandBindingPlanID = *planID;
  facts.comparePredicateKind = isComputedMask ? llvm::StringRef("sle")
                                              : llvm::StringRef();
  facts.maskRole = isComputedMask ? llvm::StringRef("predicate-mask-produced-by-compare")
                                  : llvm::StringRef();
  facts.maskSource =
      isComputedMask ? llvm::StringRef("compare-produced-mask-same-vl-scope")
                     : llvm::StringRef();
  facts.maskMemoryForm = isComputedMask ? llvm::StringRef("compare-produced-mask")
                                        : llvm::StringRef();
  facts.accumulationRouteFamilyPlanID =
      isComputedMask
          ? llvm::StringRef("rvv-computed-mask-accumulation-route-family-plan.v1")
          : llvm::StringRef();
  facts.accumulationComputeSuffix =
      isComputedMask ? llvm::StringRef("scalar-horizontal-masked-standalone-reduction")
                     : llvm::StringRef();
  facts.accumulationMaskProducerSource =
      isRuntimeScalarComputedMask ? llvm::StringRef("runtime-scalar-splat-compare-rhs")
      : isVectorComputedMask ? llvm::StringRef("vector-compare-rhs-load")
                             : llvm::StringRef();
  facts.accumulationAccumulatorContract =
      isComputedMask
          ? llvm::StringRef("scalar-seed-input-feeds-masked-horizontal-reduction")
          : llvm::StringRef();
  facts.accumulationResultContract =
      isComputedMask
          ? llvm::StringRef("scalar-horizontal-reduction-lane0-stored-to-output")
          : llvm::StringRef();
  facts.accumulationScalarCarryContract =
      isComputedMask
          ? llvm::StringRef("scalar-result-carries-across-runtime-vl-chunks")
          : llvm::StringRef();
  facts.inactiveLaneUse = inactiveUse;
  facts.inactiveLaneRequirement = inactiveRequirement;
  facts.inactiveNeutralLiteralSEW32 =
      isComputedMask
          ? getRVVSelectedBodyStandaloneReductionInactiveNeutralLiteral(
                operation, 32)
          : llvm::StringRef();
  facts.inactiveNeutralLiteralSEW64 =
      isComputedMask
          ? getRVVSelectedBodyStandaloneReductionInactiveNeutralLiteral(
                operation, 64)
          : llvm::StringRef();
  facts.reductionAccumulatorLayout =
      sew == 64 ? llvm::StringRef("scalar-i64-seed-lane0-from-accumulator-input")
      : sew == 32 ? llvm::StringRef("scalar-i32-seed-lane0-from-accumulator-input")
                  : llvm::StringRef();
  facts.reductionResultLayout =
      "store-standalone-reduction-lane0-to-output-scalar";
  facts.reductionStoreVL = "1";
  facts.scalarResultRuntimeBoundary =
      "scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1";
  if (includeRouteOperandBindingSummary) {
    if (isPlain) {
      if (isWideningStandalone)
        facts.routeOperandBindingSummary =
            (llvm::Twine(*planID) +
             ";lhs=lhs-input-buffer:lhs:abi|load|reduce-input|src-i16mf2|hdr;"
             "acc=accumulator-input-buffer:acc:abi|seed|acc-state|acc-i32|hdr;"
             "out=output-buffer:out:abi|acc-state|store|res-i32m1|hdr;"
             "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
                .str();
      else
        facts.routeOperandBindingSummary =
            (llvm::Twine(*planID) +
             ";lhs=lhs-input-buffer:lhs:abi|load|reduce-input|hdr;"
             "acc=accumulator-input-buffer:acc:abi|seed|acc-state|hdr;"
             "out=output-buffer:out:abi|acc-state|store|hdr;"
             "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
                .str();
    } else if (isVectorComputedMask) {
      facts.routeOperandBindingSummary =
          (llvm::Twine(*planID) +
           ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;"
           "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|cmp-rhs-call|hdr;"
           "src=source-input-buffer:src:abi|src-load|masked-reduce-input|" +
           inactiveUse +
           "|hdr;"
           "acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc|hdr;"
           "out=output-buffer:out:abi|acc-state|store-base|hdr;"
           "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
              .str();
    } else {
      facts.routeOperandBindingSummary =
          (llvm::Twine(*planID) +
           ";cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;"
           "rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs-call|hdr;"
           "src=source-input-buffer:src:abi|src-load|masked-reduce-input|" +
           inactiveUse +
           "|hdr;"
           "acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc|hdr;"
           "out=output-buffer:out:abi|acc-state|store-base|hdr;"
           "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")
              .str();
    }
  }

  if (sew != 0 && includeRouteOperandBindingSummary) {
    const llvm::StringRef scalarCType = sew == 64 ? llvm::StringRef("int64_t")
                                                  : llvm::StringRef("int32_t");
    const llvm::StringRef sourceScalarCType =
        isWideningStandalone ? llvm::StringRef("int16_t") : scalarCType;
    const std::string constScalarPointer =
        (llvm::Twine("const ") + scalarCType + " *").str();
    const std::string constSourcePointer =
        (llvm::Twine("const ") + sourceScalarCType + " *").str();
    const std::string mutableScalarPointer =
        (llvm::Twine(scalarCType) + " *").str();
    using support::RuntimeABIParameter;
    using support::RuntimeABIParameterOwnership;
    using support::RuntimeABIParameterRole;
    auto addParameter = [&](llvm::StringRef cName, llvm::StringRef cType,
                            RuntimeABIParameterRole role) {
      facts.runtimeABIParameters.push_back(RuntimeABIParameter(
          cName, cType, role,
          RuntimeABIParameterOwnership::TargetExportABIOwned));
    };
    if (isPlain) {
      addParameter("lhs", constSourcePointer,
                   RuntimeABIParameterRole::LHSInputBuffer);
    } else {
      addParameter("cmp_lhs", constScalarPointer,
                   RuntimeABIParameterRole::LHSInputBuffer);
      if (isRuntimeScalarComputedMask)
        addParameter("rhs_scalar", scalarCType,
                     RuntimeABIParameterRole::RHSScalarValue);
      else
        addParameter("cmp_rhs", constScalarPointer,
                     RuntimeABIParameterRole::RHSInputBuffer);
      addParameter("src", constScalarPointer,
                   RuntimeABIParameterRole::SourceInputBuffer);
    }
    addParameter("acc", constScalarPointer,
                 RuntimeABIParameterRole::AccumulatorInputBuffer);
    addParameter("out", mutableScalarPointer,
                 RuntimeABIParameterRole::OutputBuffer);
    addParameter("n", "size_t", RuntimeABIParameterRole::RuntimeElementCount);
  }

  return true;
}

std::optional<RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts>
getRVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts(
    RVVSelectedBodyOperationKind operation) {
  if (!isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFactsOperation(
          operation))
    return std::nullopt;
  return getRVVStandaloneReductionRouteFacts(operation);
}

bool isRVVFourOperandPlanID(llvm::StringRef planID) {
  return planID == kRVVCmpSelectOperandBindingPlanID ||
         planID == kRVVReduceAddOperandBindingPlanID;
}

llvm::StringRef getExpectedRVVRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  if (std::optional<llvm::StringRef> maccPlanID =
          getExpectedRVVSelectedBodyMAccRouteOperandBindingPlanID(operation))
    return *maccPlanID;
  if (std::optional<llvm::StringRef> contractionPlanID =
          getExpectedRVVSelectedBodyContractionRouteOperandBindingPlanID(
              operation))
    return *contractionPlanID;
  if (std::optional<llvm::StringRef> segment2PlanID =
          getExpectedRVVSelectedBodySegment2RouteOperandBindingPlanID(
              operation))
    return *segment2PlanID;
  if (std::optional<llvm::StringRef> elementwisePlanID =
          getExpectedRVVSelectedBodyElementwiseRouteOperandBindingPlanID(
              operation))
    return *elementwisePlanID;
  if (std::optional<llvm::StringRef> baseMemoryPlanID =
          getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingPlanID(
              operation))
    return *baseMemoryPlanID;
  if (std::optional<llvm::StringRef> standaloneReductionPlanID =
          getRVVStandaloneReductionRouteOperandBindingPlanID(operation))
    return *standaloneReductionPlanID;

  switch (operation) {
  case RVVSelectedBodyOperationKind::CmpSelect:
    return kRVVCmpSelectOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return kRVVComputedMaskSelectOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return kRVVRuntimeScalarCompareSelectOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
    return kRVVRuntimeScalarDualCompareMaskAndSelectOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::F32ClampSelect:
    return kRVVF32ClampSelectOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return kRVVDequantClampF32EpilogueOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return kRVVRuntimeScalarComputedMaskStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return kRVVRuntimeScalarComputedMaskLoadStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return kRVVRuntimeScalarComputedMaskIndexedGatherOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskIndexedScatterOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ReduceAdd:
    return kRVVReduceAddOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return kRVVComputedMaskUnitLoadStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return kRVVComputedMaskStridedStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return kRVVComputedMaskStridedLoadOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return kRVVComputedMaskIndexedGatherOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return kRVVComputedMaskIndexedScatterOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::RuntimeScalarSplatStore:
    return kRVVRuntimeScalarSplatStoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WidenI32ToI64:
    return kRVVWidenI32ToI64OperandBindingPlanID;
  case RVVSelectedBodyOperationKind::WidenI16ToI32:
    return kRVVWidenI16ToI32OperandBindingPlanID;
  case RVVSelectedBodyOperationKind::DequantizeI32ToF32:
    return kRVVDequantizeI32ToF32OperandBindingPlanID;
  default:
    break;
  }
  llvm_unreachable("unknown RVV selected-body operation kind");
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVRouteOperandBindingRole(llvm::StringRef planID,
                                      llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (std::optional<support::RuntimeABIParameterRole> elementwiseRole =
          getExpectedRVVSelectedBodyElementwiseRouteOperandBindingRole(
              planID, logicalOperand))
    return elementwiseRole;
  if (isRVVFourOperandPlanID(planID)) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (std::optional<support::RuntimeABIParameterRole> maccRole =
          getExpectedRVVSelectedBodyMAccRouteOperandBindingRole(
              planID, logicalOperand))
    return maccRole;
  if (std::optional<support::RuntimeABIParameterRole> contractionRole =
          getExpectedRVVSelectedBodyContractionRouteOperandBindingRole(
              planID, logicalOperand))
    return contractionRole;
  if (std::optional<support::RuntimeABIParameterRole> baseMemoryRole =
          getExpectedRVVSelectedBodyBaseMemoryRouteOperandBindingRole(
              planID, logicalOperand))
    return baseMemoryRole;

  if (planID == kRVVWidenI32ToI64OperandBindingPlanID ||
      planID == kRVVWidenI16ToI32OperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVDequantizeI32ToF32OperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "scale")
      return RuntimeABIParameterRole::DequantScaleValue;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVRuntimeScalarSplatStoreOperandBindingPlanID) {
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVRuntimeScalarCompareSelectOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "true_value")
      return RuntimeABIParameterRole::TrueValueInputBuffer;
    if (logicalOperand == "false_value")
      return RuntimeABIParameterRole::FalseValueInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID ==
      kRVVRuntimeScalarDualCompareMaskAndSelectOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs_a")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar_a")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "cmp_lhs_b")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "rhs_scalar_b")
      return RuntimeABIParameterRole::RHSSecondaryScalarValue;
    if (logicalOperand == "true_value")
      return RuntimeABIParameterRole::TrueValueInputBuffer;
    if (logicalOperand == "false_value")
      return RuntimeABIParameterRole::FalseValueInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVRuntimeScalarComputedMaskStoreOperandBindingPlanID ||
      planID == kRVVRuntimeScalarComputedMaskLoadStoreOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVStandaloneReductionOperandBindingPlanID ||
      planID == kRVVStandaloneReductionMinOperandBindingPlanID ||
      planID == kRVVStandaloneReductionMaxOperandBindingPlanID ||
      planID == kRVVWideningStandaloneReductionOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskStandaloneReductionOperandBindingPlanID ||
      planID == kRVVComputedMaskStandaloneReductionMinOperandBindingPlanID ||
      planID == kRVVComputedMaskStandaloneReductionMaxOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID ==
          kRVVRuntimeScalarComputedMaskStandaloneReductionOperandBindingPlanID ||
      planID ==
          kRVVRuntimeScalarComputedMaskStandaloneReductionMinOperandBindingPlanID ||
      planID ==
          kRVVRuntimeScalarComputedMaskStandaloneReductionMaxOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskUnitLoadStoreOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskStridedStoreOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (logicalOperand == "dst_stride_bytes")
      return RuntimeABIParameterRole::DestinationByteStride;
  }
  if (planID == kRVVComputedMaskStridedLoadOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (logicalOperand == "src_stride_bytes")
      return RuntimeABIParameterRole::SourceByteStride;
  }
  if (planID == kRVVComputedMaskIndexedGatherOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "index")
      return RuntimeABIParameterRole::IndexInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVRuntimeScalarComputedMaskIndexedGatherOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "index")
      return RuntimeABIParameterRole::IndexInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskIndexedScatterOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "index")
      return RuntimeABIParameterRole::IndexInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID ==
      kRVVRuntimeScalarComputedMaskIndexedScatterOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "index")
      return RuntimeABIParameterRole::IndexInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID ==
      kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "gather_src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "payload")
      return RuntimeABIParameterRole::DotRHSInputBuffer;
    if (logicalOperand == "acc")
      return RuntimeABIParameterRole::AccumulatorInputBuffer;
    if (logicalOperand == "index")
      return RuntimeABIParameterRole::IndexInputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (std::optional<support::RuntimeABIParameterRole> segment2Role =
          getExpectedRVVSelectedBodySegment2RouteOperandBindingRole(
              planID, logicalOperand))
    return segment2Role;
  if (planID == kRVVF32ClampSelectOperandBindingPlanID) {
    if (logicalOperand == "input")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "lower_bound")
      return RuntimeABIParameterRole::LowerBoundScalarValue;
    if (logicalOperand == "upper_bound")
      return RuntimeABIParameterRole::UpperBoundScalarValue;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVDequantClampF32EpilogueOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "scale")
      return RuntimeABIParameterRole::DequantScaleValue;
    if (logicalOperand == "lower_bound")
      return RuntimeABIParameterRole::LowerBoundScalarValue;
    if (logicalOperand == "upper_bound")
      return RuntimeABIParameterRole::UpperBoundScalarValue;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskSelectOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "true_value")
      return RuntimeABIParameterRole::TrueValueInputBuffer;
    if (logicalOperand == "false_value")
      return RuntimeABIParameterRole::FalseValueInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  return std::nullopt;
}

llvm::Expected<const support::RuntimeABIParameter *>
getRVVRouteOperandBindingParameter(
    const RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    llvm::StringRef materializedUse, llvm::StringRef context) {
  if (plan.planID.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " requires a non-empty RouteOperandBindingPlan");

  const RVVRouteOperandBinding *matched = nullptr;
  unsigned count = 0;
  for (const RVVRouteOperandBinding &binding : plan.bindings) {
    if (llvm::StringRef(binding.logicalOperand) != logicalOperand)
      continue;
    matched = &binding;
    ++count;
  }

  if (count != 1)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " requires exactly one logical operand '" + logicalOperand +
        "' in plan '" + plan.planID + "'");

  const bool hasUse = llvm::any_of(
      matched->materializedUses, [&](const std::string &use) {
        return llvm::StringRef(use) == materializedUse;
      });
  if (!hasUse)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " requires logical operand '" + logicalOperand +
        "' to record materialized use '" + materializedUse + "'");

  if (llvm::StringRef(matched->parameter.cName).empty() ||
      llvm::StringRef(matched->parameter.cType).empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding lookup for ") + context +
        " found empty C ABI name/type for logical operand '" +
        logicalOperand + "'");

  return &matched->parameter;
}

std::string
stringifyRVVRouteOperandBindingPlan(const RVVRouteOperandBindingPlan &plan) {
  auto stringifyUseForSummary = [&](llvm::StringRef use) -> llvm::StringRef {
    if (plan.planID ==
        "rvv-route-operand-binding:widening_product_reduce_dequant_clamp_f32.v1") {
      if (use == "src-load")
        return "ld";
      if (use == "wprod-lhs")
        return "wpl";
      if (use == "wprod-rhs")
        return "wpr";
      if (use == "src-i8mf4")
        return "i8mf4";
      if (use == "runtime-scale")
        return "scale";
      if (use == "scale-f32")
        return "f32";
      if (use == "dequant")
        return "deq";
      if (use == "runtime-lower")
        return "lo";
      if (use == "runtime-upper")
        return "up";
      if (use == "compare")
        return "cmp";
      if (use == "select")
        return "sel";
      if (use == "clamped-dequant-result")
        return "cdeq";
      if (use == "res-f32m1")
        return "f32m1";
      if (use == "setvl-avl")
        return "setvl";
      return use;
    }
    if (plan.planID ==
            "rvv-route-operand-binding:indexed_gather_unit_store.v1" ||
        plan.planID ==
            "rvv-route-operand-binding:indexed_scatter_unit_load.v1") {
      if (use == "runtime-abi-mirror")
        return "abi";
      if (use == "header-mirror")
        return "hdr";
      return use;
    }
    if (plan.planID == kRVVComputedMaskStridedStoreOperandBindingPlanID ||
        plan.planID == kRVVComputedMaskStridedLoadOperandBindingPlanID ||
        plan.planID == kRVVComputedMaskIndexedGatherOperandBindingPlanID ||
        plan.planID ==
            kRVVRuntimeScalarComputedMaskIndexedGatherOperandBindingPlanID ||
        plan.planID == kRVVComputedMaskIndexedScatterOperandBindingPlanID ||
        plan.planID ==
            kRVVRuntimeScalarComputedMaskIndexedScatterOperandBindingPlanID ||
        plan.planID ==
            kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterOperandBindingPlanID) {
      if (use == "runtime-abi-mirror" || use == "abi-mirror")
        return "abi";
      if (use == "header-mirror" || use == "hdr-mirror")
        return "hdr";
      return use;
    }
    const bool usesCompactSummary =
        plan.planID == kRVVRuntimeScalarCompareSelectOperandBindingPlanID ||
        plan.planID ==
            kRVVRuntimeScalarDualCompareMaskAndSelectOperandBindingPlanID ||
        plan.planID == kRVVF32ClampSelectOperandBindingPlanID ||
        plan.planID == kRVVDequantClampF32EpilogueOperandBindingPlanID ||
        plan.planID == kRVVComputedMaskUnitLoadStoreOperandBindingPlanID ||
        plan.planID ==
            kRVVRuntimeScalarComputedMaskStoreOperandBindingPlanID ||
        plan.planID ==
            kRVVRuntimeScalarComputedMaskLoadStoreOperandBindingPlanID;
    if (!usesCompactSummary)
      return use;
    if (plan.planID ==
        kRVVRuntimeScalarDualCompareMaskAndSelectOperandBindingPlanID) {
      if (use == "runtime-abi-mirror")
        return "abi";
      if (use == "materialized-load-base" ||
          use == "materialized-secondary-load-base" ||
          use == "materialized-true-load-base" ||
          use == "materialized-false-load-base")
        return "ld";
      if (use == "scalar-broadcast-rhs-call" ||
          use == "secondary-scalar-broadcast-rhs-call")
        return "splat";
      if (use == "compare-lhs-call" || use == "compare-rhs-call" ||
          use == "secondary-compare-lhs-call" ||
          use == "secondary-compare-rhs-call")
        return "cmp";
      if (use == "mask-and-lhs-call" || use == "mask-and-rhs-call")
        return "and";
      if (use == "select-true-call" || use == "select-false-call")
        return "sel";
      if (use == "materialized-store-base")
        return "st";
      if (use == "setvl-avl")
        return "setvl";
      if (use == "loop-control")
        return "loop";
      if (use == "header-mirror")
        return "hdr";
      return use;
    }
    if (use == "runtime-abi-mirror" || use == "abi-mirror")
      return "abi";
    if (use == "materialized-load-base")
      return "lhs-load";
    if (use == "materialized-f32-input-load-base")
      return "ld";
    if (use == "scalar-broadcast-rhs-call")
      return "splat";
    if (use == "lower-bound-splat-call")
      return "lsp";
    if (use == "upper-bound-splat-call")
      return "usp";
    if (use == "compare-lhs-call")
      return "cmp-lhs";
    if (use == "compare-rhs-call")
      return "cmp-rhs";
    if (use == "lower-compare-lhs-call" ||
        use == "lower-compare-rhs-call")
      return "lcmp";
    if (use == "upper-compare-lhs-call")
      return "ucmp";
    if (use == "materialized-secondary-load-base")
      return "lhs-b-load";
    if (use == "secondary-scalar-broadcast-rhs-call")
      return "splat-b";
    if (use == "secondary-compare-lhs-call")
      return "cmp-b-lhs";
    if (use == "secondary-compare-rhs-call")
      return "cmp-b-rhs";
    if (use == "mask-and-lhs-call")
      return "and-lhs";
    if (use == "mask-and-rhs-call")
      return "and-rhs";
    if (use == "select-mask-call")
      return "sel-mask";
    if (use == "materialized-true-load-base")
      return "true-load";
    if (use == "select-true-call")
      return "sel-true";
    if (use == "materialized-false-load-base")
      return "false-load";
    if (use == "select-false-call")
      return "sel-false";
    if (use == "lower-select-true-call")
      return "lselT";
    if (use == "lower-select-false-call")
      return "lselF";
    if (use == "upper-select-true-call")
      return "uselT";
    if (use == "materialized-store-base")
      return "store";
    if (use == "materialized-source-load-base")
      return "src-load";
    if (use == "dequant-src")
      return "deq-src";
    if (use == "runtime-scale")
      return "scale";
    if (use == "scale-f32")
      return "sf32";
    if (use == "src-i32m1")
      return "i32m1";
    if (use == "relation-signed-i32m1-to-f32m1-scale-f32")
      return "i32-f32-scale";
    if (use == "masked-store-source-call")
      return "mstore-src";
    if (use == "materialized-masked-store-base")
      return "mstore-base";
    if (use == "masked-store-destination-call")
      return "mstore-dst";
    if (use == "materialized-masked-load-base")
      return "mload-base";
    if (use == "masked-load-source-call")
      return "mload-src";
    if (use == "materialized-old-destination-load-base")
      return "old-dst-load";
    if (use == "masked-load-passthrough-call")
      return "mload-pass";
    if (use == "setvl-avl")
      return "setvl";
    if (use == "loop-control")
      return "loop";
    if (use == "header-mirror")
      return "hdr";
    return use;
  };

  std::string text;
  llvm::raw_string_ostream os(text);
  os << plan.planID;
  for (const RVVRouteOperandBinding &binding : plan.bindings) {
    os << ";" << binding.logicalOperand << "="
       << support::stringifyRuntimeABIParameterRole(binding.parameter.role)
       << ":" << binding.parameter.cName << ":";
    for (auto indexedUse : llvm::enumerate(binding.materializedUses)) {
      if (indexedUse.index() != 0)
        os << "|";
      os << stringifyUseForSummary(indexedUse.value());
    }
  }
  os.flush();
  return text;
}

std::string stringifyRVVRuntimeABIParametersForDiagnostic(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
  std::string text;
  llvm::raw_string_ostream os(text);
  bool first = true;
  for (const support::RuntimeABIParameter &parameter : parameters) {
    if (!first)
      os << ",";
    first = false;
    os << parameter.cName << ":" << parameter.cType << ":"
       << support::stringifyRuntimeABIParameterRole(parameter.role) << ":"
       << support::stringifyRuntimeABIParameterOwnership(parameter.ownership);
  }
  os.flush();
  return text;
}

llvm::Error verifyRVVRouteOperandBindingPlan(
    const RVVRouteOperandBindingPlan &plan, llvm::StringRef expectedPlanID,
    llvm::StringRef expectedRuntimeABIOrder, llvm::StringRef context) {
  if (llvm::StringRef(plan.planID) != expectedPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires plan id '" + expectedPlanID + "' but found '" +
        plan.planID + "'");
  if (plan.bindings.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires at least one binding");

  llvm::StringSet<> logicalOperands;
  llvm::StringSet<> runtimeRoles;
  std::string runtimeABIOrder;
  llvm::raw_string_ostream orderOS(runtimeABIOrder);
  bool firstBinding = true;
  for (const RVVRouteOperandBinding &binding : plan.bindings) {
    if (binding.logicalOperand.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found empty logical operand");
    if (!logicalOperands.insert(binding.logicalOperand).second)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found duplicate logical operand '" +
          binding.logicalOperand + "'");
    llvm::StringRef role =
        support::stringifyRuntimeABIParameterRole(binding.parameter.role);
    std::optional<support::RuntimeABIParameterRole> expectedRole =
        getExpectedRVVRouteOperandBindingRole(plan.planID,
                                              binding.logicalOperand);
    if (!expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found unsupported logical operand '" +
          binding.logicalOperand + "' in plan '" + plan.planID + "'");
    if (binding.parameter.role != *expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " requires logical operand '" + binding.logicalOperand +
          "' to bind runtime ABI role '" +
          support::stringifyRuntimeABIParameterRole(*expectedRole) +
          "' but found '" + role + "'");
    if (!runtimeRoles.insert(role).second)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " found duplicate runtime ABI role '" + role + "'");
    if (binding.parameter.cName.empty() || binding.parameter.cType.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " requires non-empty C ABI name/type for logical operand '" +
          binding.logicalOperand + "'");
    if (binding.materializedUses.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          context + " requires materialized uses for logical operand '" +
          binding.logicalOperand + "'");
    if (!firstBinding)
      orderOS << ",";
    firstBinding = false;
    orderOS << binding.parameter.cName;
  }
  orderOS.flush();
  if (runtimeABIOrder != expectedRuntimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires runtime ABI mirror order '" +
        expectedRuntimeABIOrder + "' but plan records '" + runtimeABIOrder +
        "'");

  return llvm::Error::success();
}

llvm::Error verifyRVVRouteOperandBindingClosure(
    const RVVRouteOperandBindingPlan &plan,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  llvm::StringRef expectedPlanID =
      getExpectedRVVRouteOperandBindingPlanID(description.operation);
  if (std::optional<RVVWideningProductRouteFacts> wideningProductFacts =
          getRVVWideningProductRouteFacts(description))
    expectedPlanID = wideningProductFacts->routeOperandBindingPlanID;
  if (expectedPlanID.empty()) {
    if (!plan.planID.empty() || !description.routeOperandBindingPlanID.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding closure for ") + context +
          " found a binding plan for a route without binding support");
    if (!description.routeOperandBindingSummary.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding closure for ") + context +
          " found a binding mirror summary for a route without binding "
          "support");
    return llvm::Error::success();
  }

  if (llvm::StringRef(plan.planID) != expectedPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding closure for ") + context +
        " requires plan id '" + expectedPlanID + "' but found '" +
        plan.planID + "'");
  if (llvm::StringRef(description.routeOperandBindingPlanID) != expectedPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding closure for ") + context +
        " requires route description mirror plan id '" + expectedPlanID +
        "' but found '" + description.routeOperandBindingPlanID + "'");
  if (llvm::Error error = verifyRVVRouteOperandBindingPlan(
          plan, expectedPlanID, description.runtimeABIOrder, context))
    return error;

  llvm::SmallVector<support::RuntimeABIParameter, 8> planParameters;
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    planParameters.push_back(binding.parameter);
  if (!support::runtimeABIParametersEqual(planParameters,
                                          description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding closure for ") + context +
        " requires runtime ABI parameter mirrors to match the binding plan; "
        "plan has '" +
        stringifyRVVRuntimeABIParametersForDiagnostic(planParameters) +
        "' while description has '" +
        stringifyRVVRuntimeABIParametersForDiagnostic(
            description.runtimeABIParameters) +
        "'");

  std::string expectedSummary = stringifyRVVRouteOperandBindingPlan(plan);
  if (description.routeOperandBindingSummary != expectedSummary)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding closure for ") + context +
        " requires route operand binding mirror summary '" + expectedSummary +
        "' but found '" + description.routeOperandBindingSummary + "'");
  return llvm::Error::success();
}

namespace {

constexpr llvm::StringLiteral kRVVRuntimeAVLVLControlPlanID(
    "rvv-runtime-avl-vl-control-plan.v1");

llvm::Error makeRVVRuntimeAVLVLControlPlanError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV runtime AVL/VL control plan invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::StringRef stringifyRuntimeControlTailPolicy(
    tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
}

llvm::StringRef stringifyRuntimeControlMaskPolicy(
    tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  return "unknown";
}

llvm::Expected<support::RuntimeABIParameter>
getRuntimeAVLParameterBindingFromValue(mlir::Value value,
                                       llvm::StringRef context) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " AVL must be defined by explicit tcrv_rvv.runtime_abi_value");
  if (llvm::Error error =
          verifyRVVRuntimeABIValueRoleOpInterface(binding.getOperation()))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL binding carries unsupported runtime ABI "
                              "role '" +
        binding.getRole() + "'");
  if (*role != support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " AVL must bind the single runtime-element-count ABI role; got '" +
        binding.getRole() + "'");

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL binding carries unsupported runtime ABI "
                              "ownership '" +
        binding.getOwnership() + "'");

  support::RuntimeABIParameter parameter(binding.getCName(),
                                         binding.getCType(), *role,
                                         *ownership);
  if (parameter.cName != tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName())
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL runtime ABI parameter must be named '" +
        tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName() +
        "' but found '" + parameter.cName + "'");
  if (parameter.cType != "size_t")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " AVL runtime ABI parameter must have C type "
                              "'size_t' but found '" +
        parameter.cType + "'");
  if (parameter.ownership !=
      support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " AVL runtime ABI parameter must be target-export ABI owned");

  return parameter;
}

llvm::Error validateSingleRuntimeAVLBinding(tcrv::exec::VariantOp variant,
                                            mlir::Value runtimeAVLValue,
                                            llvm::StringRef context) {
  if (!variant)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires a materialized selected tcrv.exec.variant");

  auto selectedBinding =
      runtimeAVLValue.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!selectedBinding)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " runtime AVL value must be a tcrv_rvv.runtime_abi_value");

  unsigned runtimeElementCountBindings = 0;
  bool selectedIsTheRuntimeCount = false;
  variant.getBody().walk([&](tcrv::rvv::RuntimeABIValueOp op) {
    std::optional<support::RuntimeABIParameterRole> role =
        support::symbolizeRuntimeABIParameterRole(op.getRole());
    if (!role || *role != support::RuntimeABIParameterRole::RuntimeElementCount)
      return;
    ++runtimeElementCountBindings;
    if (op.getOperation() == selectedBinding.getOperation())
      selectedIsTheRuntimeCount = true;
  });

  if (runtimeElementCountBindings != 1)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires exactly one runtime-element-count ABI binding in the "
        "selected RVV variant");
  if (!selectedIsTheRuntimeCount)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " setvl AVL must consume the selected variant's single runtime n ABI "
        "binding");
  return llvm::Error::success();
}

llvm::Expected<RVVRuntimeAVLVLControlPlan>
buildRuntimeAVLVLControlPlan(
    tcrv::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy,
    llvm::StringRef runtimeABIOrder, llvm::StringRef context) {
  llvm::Expected<support::RuntimeABIParameter> runtimeAVL =
      getRuntimeAVLParameterBindingFromValue(runtimeAVLValue, context);
  if (!runtimeAVL)
    return runtimeAVL.takeError();
  if (llvm::Error error =
          validateSingleRuntimeAVLBinding(variant, runtimeAVLValue, context))
    return std::move(error);

  const tcrv::rvv::RVVSelectedBodyConfigVLContract &configContract =
      tcrv::rvv::getRVVSelectedBodyConfigVLContract(sew, lmul, policy);

  RVVRuntimeAVLVLControlPlan plan;
  plan.sew = sew;
  plan.lmul = lmul;
  plan.policy = policy;
  if (policy) {
    plan.tailPolicy = stringifyRuntimeControlTailPolicy(policy.getTail());
    plan.maskPolicy = stringifyRuntimeControlMaskPolicy(policy.getMask());
  }
  plan.controlPlanID = kRVVRuntimeAVLVLControlPlanID;
  plan.configContractID = configContract.configContractID;
  plan.runtimeVLContractID = configContract.runtimeVLContractID;
  plan.runtimeAVLABIParameterName =
      configContract.runtimeAVLABIParameterName;
  plan.runtimeAVLASource = configContract.runtimeAVLASource;
  plan.runtimeABIOrder = runtimeABIOrder;
  plan.vlDefOpName = configContract.vlDefOpName;
  plan.vlScopeOpName = configContract.vlScopeOpName;
  plan.vlUses = configContract.vlUses;
  plan.emitCLoopKind = configContract.emitCLoopKind;
  plan.emitCLoopInductionName = configContract.emitCLoopInductionName;
  plan.emitCFullChunkVLName = configContract.emitCFullChunkVLName;
  plan.emitCLoopVLName = tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName();
  plan.remainingAVLMetadata = configContract.remainingAVLMetadata;
  plan.pointerAdvanceMetadata = configContract.pointerAdvanceMetadata;
  plan.boundedSlice = configContract.boundedSlice;
  plan.multiVL = configContract.multiVL;
  plan.runtimeAVLValue = runtimeAVLValue;
  plan.runtimeAVLParameter = std::move(*runtimeAVL);

  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(plan, context))
    return std::move(error);
  return plan;
}

} // namespace

llvm::StringRef getRVVRuntimeAVLVLControlPlanID() {
  return kRVVRuntimeAVLVLControlPlanID;
}

std::optional<RVVRuntimeAVLVLSelectedBoundaryContract>
getRVVRuntimeAVLVLSelectedBoundaryContract(
    std::int64_t sew, llvm::StringRef lmul, llvm::StringRef tailPolicy,
    llvm::StringRef maskPolicy, llvm::StringRef configContractID,
    llvm::StringRef setVLIntrinsic, llvm::StringRef vlCType,
    llvm::StringRef runtimeABIOrder,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    llvm::StringRef consumerLabel) {
  const support::RuntimeABIParameter *runtimeAVLParameter = nullptr;
  for (const support::RuntimeABIParameter &parameter : runtimeABIParameters) {
    if (parameter.role != support::RuntimeABIParameterRole::RuntimeElementCount)
      continue;
    if (runtimeAVLParameter)
      return std::nullopt;
    runtimeAVLParameter = &parameter;
  }
  if (!runtimeAVLParameter)
    return std::nullopt;

  const tcrv::rvv::RVVSelectedBodyConfigVLContract &configContract =
      tcrv::rvv::getRVVSelectedBodyConfigVLContract(sew, lmul);

  RVVRuntimeAVLVLSelectedBoundaryContract contract;
  contract.consumerLabel = consumerLabel;
  contract.sew = sew;
  contract.lmul = lmul.str();
  contract.tailPolicy = tailPolicy.str();
  contract.maskPolicy = maskPolicy.str();
  contract.configContractID = configContractID.str();
  contract.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID().str();
  contract.runtimeVLContractID = configContract.runtimeVLContractID.str();
  contract.runtimeAVLABIParameterName =
      configContract.runtimeAVLABIParameterName.str();
  contract.runtimeAVLASource = configContract.runtimeAVLASource.str();
  contract.runtimeABIOrder = runtimeABIOrder.str();
  contract.selectedBoundaryOpName = configContract.vlScopeOpName.str();
  contract.selectedBodyProvenance =
      "provider-derived selected typed tcrv_rvv setvl/with_vl body";
  contract.vlDefOpName = configContract.vlDefOpName.str();
  contract.vlScopeOpName = configContract.vlScopeOpName.str();
  contract.vlUses = configContract.vlUses.str();
  contract.setVLIntrinsic = setVLIntrinsic.str();
  contract.vlCType = vlCType.str();
  contract.emitCLoopKind = configContract.emitCLoopKind.str();
  contract.emitCLoopInductionName =
      configContract.emitCLoopInductionName.str();
  contract.emitCFullChunkVLName = configContract.emitCFullChunkVLName.str();
  contract.emitCLoopVLName = tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName().str();
  contract.remainingAVLMetadata = configContract.remainingAVLMetadata.str();
  contract.pointerAdvanceMetadata =
      configContract.pointerAdvanceMetadata.str();
  contract.boundedSlice = configContract.boundedSlice.str();
  contract.multiVL = configContract.multiVL.str();
  contract.runtimeAVLParameter = *runtimeAVLParameter;
  return contract;
}


llvm::Error verifyRVVRuntimeAVLVLControlPlan(
    const RVVRuntimeAVLVLControlPlan &plan, llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVRuntimeAVLVLControlPlanError(
        "verification requires a non-empty context");
  if (plan.controlPlanID != kRVVRuntimeAVLVLControlPlanID)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must use runtime control plan '" +
        kRVVRuntimeAVLVLControlPlanID + "'");
  if (!plan.policy)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " requires explicit RVV policy");
  // The deferred-wide (N3) product-reduce-dequant route runs its runtime AVL/VL
  // loop at the wide strip config (sew8/m2) -- the structural setvl config of
  // the i8m2 -> i16m4 -> i32m8 winner. Admit that parallel config alongside the
  // first-slice dataflow configs (I5: it is the realized setvl config).
  if (!tcrv::rvv::isRVVFirstSliceDataflowConfig(plan.sew, plan.lmul) &&
      !tcrv::rvv::isRVVDeferredWideStripConfig(plan.sew, plan.lmul))
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires a supported typed RVV SEW/LMUL config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(plan.policy) &&
      !tcrv::rvv::isRVVUndisturbedPolicy(plan.policy))
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires an explicitly supported runtime VL policy");
  if (plan.runtimeAVLABIParameterName !=
      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName())
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must use runtime AVL ABI parameter '" +
        tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName() + "'");
  if (plan.runtimeAVLASource != "runtime_abi:n")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must derive runtime AVL source from runtime_abi:n");
  if (plan.runtimeAVLParameter.role !=
      support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must carry runtime-element-count as AVL parameter role");
  if (plan.runtimeAVLParameter.cName !=
      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName())
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must carry runtime AVL ABI parameter name '" +
        tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName() + "'");
  if (plan.runtimeAVLParameter.cType != "size_t")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must carry runtime AVL ABI C type 'size_t'");
  if (plan.runtimeAVLParameter.ownership !=
      support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must carry target-export-owned runtime AVL ABI ownership");
  if (plan.vlDefOpName != "tcrv_rvv.setvl" ||
      plan.vlScopeOpName != "tcrv_rvv.with_vl")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must route VL through tcrv_rvv.setvl and tcrv_rvv.with_vl");
  if (plan.remainingAVLMetadata != "n-offset" ||
      plan.pointerAdvanceMetadata != "offset")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " must carry remaining AVL 'n-offset' and pointer advance 'offset'");
  if (plan.multiVL != "supported")
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " must keep multi-VL support validated");
  return llvm::Error::success();
}

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
    tcrv::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy,
    llvm::StringRef runtimeABIOrder, llvm::StringRef context) {
  return buildRuntimeAVLVLControlPlan(variant, runtimeAVLValue, sew, lmul,
                                      policy, runtimeABIOrder, context);
}

llvm::Expected<RVVRuntimeAVLVLControlPlan>
deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
    tcrv::exec::VariantOp variant, tcrv::rvv::SetVLOp setvl,
    tcrv::rvv::WithVLOp withVL, llvm::StringRef runtimeABIOrder,
    llvm::StringRef context) {
  if (!setvl)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " requires a validated tcrv_rvv.setvl op");
  if (!withVL)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " requires a validated tcrv_rvv.with_vl op");
  if (setvl->getBlock() != withVL->getBlock() ||
      !setvl->isBeforeInBlock(withVL.getOperation()))
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) +
        " requires tcrv_rvv.setvl to dominate the tcrv_rvv.with_vl scope");

  tcrv::rvv::RVVConfigContractDiagnostic structure =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(setvl, withVL);
  if (!structure.ok)
    return makeRVVRuntimeAVLVLControlPlanError(
        llvm::Twine(context) + " " + structure.message);

  tcrv::rvv::RVVCompileTimeConfig config =
      tcrv::rvv::getRVVSetVLCompileTimeConfig(setvl);
  return buildRuntimeAVLVLControlPlan(variant, setvl.getAvl(), config.sew,
                                      config.lmul, config.policy,
                                      runtimeABIOrder, context);
}

std::optional<RVVWideningConversionRouteFacts>
getRVVWideningConversionRouteFacts(RVVSelectedBodyOperationKind operation) {
  return buildRVVWideningConversionRouteFacts(operation);
}

std::optional<RVVDequantizationRouteFacts>
getRVVDequantizationRouteFacts(RVVSelectedBodyOperationKind operation) {
  return buildRVVDequantizationRouteFacts(operation);
}











std::optional<RVVUnitStrideMaskedMemoryRouteFacts>
getRVVUnitStrideMaskedMemoryRouteFacts(RVVSelectedBodyOperationKind operation) {
  return getRVVUnitStrideMaskedMemoryRouteFacts(
      operation, tcrv::rvv::getRVVFirstSliceSEWBits(),
      tcrv::rvv::getRVVLMULM1());
}

std::optional<RVVUnitStrideMaskedMemoryRouteFacts>
getRVVUnitStrideMaskedMemoryRouteFacts(RVVSelectedBodyOperationKind operation,
                                       std::int64_t sew,
                                       llvm::StringRef lmul) {
  const bool isStaticMaskLoad =
      operation == RVVSelectedBodyOperationKind::MaskedUnitLoadStore;
  const bool isStaticMaskStore =
      operation == RVVSelectedBodyOperationKind::MaskedUnitStore;
  const bool isRuntimeScalarStore =
      operation == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
  const bool isRuntimeScalarLoadStore =
      operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
  const bool isComputedMaskUnitLoadStore =
      operation == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore;
  const bool isStaticMask = isStaticMaskLoad || isStaticMaskStore;
  const bool isRuntimeScalar =
      isRuntimeScalarStore || isRuntimeScalarLoadStore;
  const bool isComputedMask =
      isRuntimeScalar || isComputedMaskUnitLoadStore;
  if (!isStaticMask && !isComputedMask)
    return std::nullopt;
  if (isStaticMask || isComputedMaskUnitLoadStore) {
    if (sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        lmul != tcrv::rvv::getRVVLMULM1())
      return std::nullopt;
  } else if (!isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(sew,
                                                                     lmul)) {
    return std::nullopt;
  }

  RVVUnitStrideMaskedMemoryRouteFacts facts;
  facts.operation = operation;
  facts.sew = sew;
  facts.lmul = lmul;
  facts.vlCType = "size_t";
  facts.vectorTypeName = getRVVSelectedBodyVectorTypeName(sew, lmul);
  facts.vectorCType = getRVVSelectedBodySignedVectorCType(sew, lmul);
  facts.maskTypeName = getRVVSelectedBodyMaskTypeName(sew, lmul);
  facts.maskCType = getRVVSelectedBodyMaskCType(sew, lmul);
  facts.scalarCType =
      isRuntimeScalar ? getRVVRuntimeScalarComputedMaskMemoryElementCType(sew,
                                                                          lmul)
                      : llvm::StringRef();
  facts.setVLIntrinsic = getRVVSelectedBodySetVLIntrinsic(sew, lmul);
  facts.vectorLoadIntrinsic = getRVVSelectedBodyVectorLoadIntrinsic(sew, lmul);
  facts.maskedLoadIntrinsic =
      (isStaticMaskLoad || isRuntimeScalarLoadStore ||
       isComputedMaskUnitLoadStore)
          ? getRVVSelectedBodyMaskedLoadIntrinsic(sew, lmul)
          : llvm::StringRef();
  facts.storeIntrinsic =
      (isStaticMaskStore || isRuntimeScalarStore)
          ? getRVVSelectedBodyMaskedStoreIntrinsic(sew, lmul)
          : getRVVSelectedBodyStoreIntrinsic(sew, lmul);
  facts.rhsScalarSplatIntrinsic =
      isRuntimeScalar ? getRVVSelectedBodyScalarSplatIntrinsic(sew, lmul)
                      : llvm::StringRef();
  facts.comparePredicateKind =
      isRuntimeScalar ? llvm::StringRef("sle")
      : isComputedMaskUnitLoadStore ? llvm::StringRef("slt")
                                    : llvm::StringRef();
  facts.compareIntrinsic =
      isStaticMask
          ? getRVVSelectedBodyMaskFromI32Intrinsic(lmul)
          : getRVVSelectedBodyCompareIntrinsicForPredicate(
                facts.comparePredicateKind, sew, lmul);
  facts.maskRole = isStaticMask
                       ? llvm::StringRef(kRVVMaskRole)
                       : llvm::StringRef(kRVVMaskedPredicateMaskRole);
  facts.maskSource = isStaticMask
                         ? llvm::StringRef(kRVVMaskSource)
                         : llvm::StringRef(kRVVMaskedCompareMaskSource);
  facts.maskMemoryForm =
      isStaticMask ? llvm::StringRef(kRVVMaskMemoryForm)
                   : llvm::StringRef(kRVVComputedMaskMemoryMaskMemoryForm);

  if (isStaticMask) {
    std::optional<RVVBaseMemoryMovementRouteFacts> baseFacts =
        getRVVBaseMemoryMovementRouteFacts(operation);
    if (!baseFacts)
      return std::nullopt;
    facts.memoryForm = baseFacts->memoryForm;
    facts.tailPolicy = baseFacts->tailPolicy;
    facts.maskPolicy = baseFacts->maskPolicy;
    facts.runtimeControlPlanID = baseFacts->runtimeControlPlanID;
    facts.runtimeABIOrder = baseFacts->runtimeABIOrder;
    facts.targetLeafProfile = baseFacts->targetLeafProfile;
    facts.providerSupportedMirror = baseFacts->providerSupportedMirror;
    facts.requiredHeaderDeclarations = baseFacts->requiredHeaderDeclarations;
    facts.cTypeMappingSummary = baseFacts->cTypeMappingSummary;
    facts.routeOperandBindingPlanID = baseFacts->routeOperandBindingPlanID;
    facts.baseMemoryMovementRouteFamilyPlanID = baseFacts->routeFamilyPlanID;
    facts.typedComputeOpName =
        isStaticMaskLoad ? llvm::StringRef("tcrv_rvv.masked_load")
                         : llvm::StringRef("tcrv_rvv.masked_store");
    facts.inactiveLaneContract = baseFacts->inactiveLaneContract;
    facts.maskedPassthroughLayout = baseFacts->maskedPassthroughLayout;
    facts.maskedMemoryLayout = baseFacts->indexedMemoryLayout;
    facts.sourceMemoryForm = baseFacts->sourceMemoryForm;
    facts.destinationMemoryForm = baseFacts->destinationMemoryForm;
    facts.routeOperandBindingSummary = baseFacts->routeOperandBindingSummary;
    facts.logicalOperands = baseFacts->logicalOperands;
    facts.runtimeABIParameters = baseFacts->runtimeABIParameters;
    return facts;
  }

  facts.memoryForm = getComputedMaskMemoryRouteFamilyMemoryForm(operation);
  facts.tailPolicy = isRuntimeScalarStore ? llvm::StringRef("undisturbed")
                                          : llvm::StringRef("agnostic");
  facts.maskPolicy = isRuntimeScalarStore ? llvm::StringRef("undisturbed")
                                          : llvm::StringRef("agnostic");
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getComputedMaskMemoryRuntimeABIOrder(operation);
  facts.targetLeafProfile = getComputedMaskMemoryTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      getComputedMaskMemoryProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations =
      getComputedMaskMemoryRequiredHeaderDeclarations(operation);
  facts.cTypeMappingSummary = getComputedMaskMemoryCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID =
      getExpectedRVVRouteOperandBindingPlanID(operation);
  facts.computedMaskMemoryRouteFamilyPlanID =
      kRVVComputedMaskMemoryRouteFamilyPlanID;
  facts.computedMaskMemoryMaskProducerSource =
      getComputedMaskMemoryProducerSource(operation);
  facts.maskTailPolicyRouteFamilyPlanID = kRVVMaskTailPolicyRouteFamilyPlanID;
  facts.maskTailPolicyOwner = kRVVComputedMaskMemoryMaskTailPolicyOwner;
  facts.typedComputeOpName =
      (isRuntimeScalarStore)
          ? llvm::StringRef("tcrv_rvv.masked_store")
          : llvm::StringRef("tcrv_rvv.masked_load");
  facts.inactiveLaneContract = getComputedMaskMemoryInactiveLaneContract(operation);
  facts.maskedPassthroughLayout =
      getComputedMaskMemoryPassthroughLayout(operation);
  facts.maskedMemoryLayout = getComputedMaskMemoryLayout(operation);
  facts.sourceMemoryForm = getComputedMaskMemorySourceMemoryForm(operation);
  facts.destinationMemoryForm =
      getComputedMaskMemoryDestinationMemoryForm(operation);
  facts.runtimeABIParameters =
      isRuntimeScalar
          ? tcrv::rvv::
                buildRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters(
                    facts.scalarCType)
          : tcrv::rvv::getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();

  RVVRouteOperandBindingPlan plan =
      buildUnitStrideMaskedMemoryRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary = stringifyRVVRouteOperandBindingPlan(plan);
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    facts.logicalOperands.push_back(binding.logicalOperand);

  return facts;
}







std::optional<RVVComputedMaskIndexedMemoryRouteFacts>
getRVVComputedMaskIndexedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation) {
  const bool isGather =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isRuntimeScalarGather =
      operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
  const bool isScatter =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarScatter =
      operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
  const bool isRuntimeScalarGatherMAccScatter =
      operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  if (!isGather && !isRuntimeScalarGather && !isScatter &&
      !isRuntimeScalarScatter && !isRuntimeScalarGatherMAccScatter)
    return std::nullopt;

  RVVComputedMaskIndexedMemoryRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = getComputedMaskMemoryRouteFamilyMemoryForm(operation);
  facts.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
  facts.lmul = tcrv::rvv::getRVVLMULM1();
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getComputedMaskMemoryRuntimeABIOrder(operation);
  facts.targetLeafProfile = getComputedMaskMemoryTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      getComputedMaskMemoryProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations =
      getComputedMaskMemoryRequiredHeaderDeclarations(operation);
  facts.cTypeMappingSummary =
      getComputedMaskMemoryCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID =
      getExpectedRVVRouteOperandBindingPlanID(operation);
  facts.typedComputeOpName =
      isRuntimeScalarGatherMAccScatter
          ? llvm::StringRef(kRVVCompositeGatherMAccScatterTypedComputeChain)
      : (isGather || isRuntimeScalarGather)
          ? llvm::StringRef("tcrv_rvv.masked_indexed_load")
          : llvm::StringRef("tcrv_rvv.masked_indexed_store");
  facts.comparePredicateKind =
      getComputedMaskMemoryComparePredicateKind(operation);
  facts.vlCType = "size_t";
  facts.vectorTypeName = getRVVSelectedBodyVectorTypeName(facts.sew,
                                                          facts.lmul);
  facts.vectorCType = getRVVSelectedBodySignedVectorCType(facts.sew,
                                                          facts.lmul);
  facts.indexVectorTypeName =
      getRVVSelectedBodyIndexVectorTypeName(facts.sew, facts.lmul);
  facts.indexVectorCType =
      getRVVSelectedBodyIndexVectorCType(facts.sew, facts.lmul);
  facts.maskTypeName = getRVVSelectedBodyMaskTypeName(facts.sew, facts.lmul);
  facts.maskCType = getRVVSelectedBodyMaskCType(facts.sew, facts.lmul);
  facts.setVLIntrinsic = getRVVSelectedBodySetVLIntrinsic(facts.sew,
                                                          facts.lmul);
  facts.vectorLoadIntrinsic =
      getRVVSelectedBodyVectorLoadIntrinsic(facts.sew, facts.lmul);
  facts.indexLoadIntrinsic =
      getRVVSelectedBodyIndexLoadIntrinsic(facts.sew, facts.lmul);
  facts.indexScaleIntrinsic =
      getRVVSelectedBodyIndexScaleIntrinsic(facts.sew, facts.lmul);
  facts.maskedIndexedLoadIntrinsic =
      (isGather || isRuntimeScalarGather || isRuntimeScalarGatherMAccScatter)
          ? getRVVSelectedBodyMaskedIndexedLoadIntrinsic(facts.sew, facts.lmul)
          : llvm::StringRef();
  facts.maskedIndexedStoreIntrinsic =
      (isScatter || isRuntimeScalarScatter || isRuntimeScalarGatherMAccScatter)
          ? getRVVSelectedBodyMaskedIndexedStoreIntrinsic(facts.sew,
                                                          facts.lmul)
          : llvm::StringRef();
  facts.maskedStoreIntrinsic =
      (isGather || isRuntimeScalarGather)
          ? getRVVSelectedBodyStoreIntrinsic(facts.sew, facts.lmul)
          : llvm::StringRef();
  facts.rhsScalarSplatIntrinsic =
      (isRuntimeScalarGather || isRuntimeScalarScatter ||
       isRuntimeScalarGatherMAccScatter)
          ? getRVVSelectedBodyScalarSplatIntrinsic(facts.sew, facts.lmul)
          : llvm::StringRef();
  facts.compareIntrinsic = getRVVSelectedBodyCompareIntrinsicForPredicate(
      facts.comparePredicateKind, facts.sew, facts.lmul);
  facts.computedMaskMemoryRouteFamilyPlanID =
      kRVVComputedMaskMemoryRouteFamilyPlanID;
  facts.computedMaskMemoryMaskProducerSource =
      getComputedMaskMemoryProducerSource(operation);
  facts.maskTailPolicyRouteFamilyPlanID = kRVVMaskTailPolicyRouteFamilyPlanID;
  facts.maskTailPolicyOwner = kRVVComputedMaskMemoryMaskTailPolicyOwner;
  facts.maskRole = kRVVMaskedPredicateMaskRole;
  facts.maskSource = kRVVMaskedCompareMaskSource;
  facts.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  facts.inactiveLaneContract =
      getComputedMaskMemoryInactiveLaneContract(operation);
  facts.maskedPassthroughLayout =
      getComputedMaskMemoryPassthroughLayout(operation);
  facts.indexedMemoryLayout = getComputedMaskMemoryLayout(operation);
  facts.indexedWriteSideContract =
      isRuntimeScalarGatherMAccScatter
          ? llvm::StringRef(
                kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterWriteSideContract)
      : (isScatter || isRuntimeScalarScatter)
          ? llvm::StringRef(kRVVComputedMaskIndexedScatterWriteSideContract)
          : llvm::StringRef();
  facts.sourceMemoryForm = getComputedMaskMemorySourceMemoryForm(operation);
  facts.destinationMemoryForm =
      getComputedMaskMemoryDestinationMemoryForm(operation);
  facts.indexEEW = 32;
  facts.offsetUnit = kRVVIndexedGatherOffsetUnit;
  facts.indexSource = kRVVIndexSource;
  facts.indexUniqueness =
      (isScatter || isRuntimeScalarScatter || isRuntimeScalarGatherMAccScatter)
          ? llvm::StringRef(kRVVIndexedScatterIndexUniqueness)
          : llvm::StringRef();
  facts.indexedDataMemoryForm =
      (isGather || isRuntimeScalarGather || isRuntimeScalarGatherMAccScatter)
          ? llvm::StringRef(kRVVMaskedIndexedLoadSourceMemoryForm)
          : llvm::StringRef();
  facts.indexedDestinationMemoryForm =
      (isScatter || isRuntimeScalarScatter || isRuntimeScalarGatherMAccScatter)
          ? llvm::StringRef(kRVVMaskedIndexedStoreDestinationMemoryForm)
          : llvm::StringRef();
  facts.runtimeABIParameters =
      getRVVComputedMaskIndexedMemoryRuntimeABIParameters(operation);

  RVVRouteOperandBindingPlan plan =
      buildComputedMaskIndexedMemoryRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    facts.logicalOperands.push_back(binding.logicalOperand);

  return facts;
}








std::optional<RVVComputedMaskStridedMemoryRouteFacts>
getRVVComputedMaskStridedMemoryRouteFacts(
    RVVSelectedBodyOperationKind operation) {
  const bool isStore =
      operation == RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
  const bool isLoad =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
  if (!isStore && !isLoad)
    return std::nullopt;

  RVVComputedMaskStridedMemoryRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = getComputedMaskMemoryRouteFamilyMemoryForm(operation);
  facts.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
  facts.lmul = tcrv::rvv::getRVVLMULM1();
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getComputedMaskMemoryRuntimeABIOrder(operation);
  facts.targetLeafProfile = getComputedMaskMemoryTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      getComputedMaskMemoryProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations =
      getComputedMaskMemoryRequiredHeaderDeclarations(operation);
  facts.cTypeMappingSummary =
      getComputedMaskMemoryCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID =
      getExpectedRVVRouteOperandBindingPlanID(operation);
  facts.typedComputeOpName =
      isStore ? llvm::StringRef("tcrv_rvv.masked_strided_store")
              : llvm::StringRef("tcrv_rvv.masked_strided_load");
  facts.comparePredicateKind = getComputedMaskMemoryComparePredicateKind(
      operation);
  facts.vlCType = "size_t";
  facts.vectorTypeName = getRVVSelectedBodyVectorTypeName(facts.sew,
                                                          facts.lmul);
  facts.vectorCType = getRVVSelectedBodySignedVectorCType(facts.sew,
                                                          facts.lmul);
  facts.maskTypeName = getRVVSelectedBodyMaskTypeName(facts.sew, facts.lmul);
  facts.maskCType = getRVVSelectedBodyMaskCType(facts.sew, facts.lmul);
  facts.setVLIntrinsic = getRVVSelectedBodySetVLIntrinsic(facts.sew,
                                                          facts.lmul);
  facts.vectorLoadIntrinsic =
      getRVVSelectedBodyVectorLoadIntrinsic(facts.sew, facts.lmul);
  facts.maskedLoadIntrinsic =
      isLoad ? getRVVSelectedBodyMaskedStridedLoadIntrinsic(facts.sew,
                                                            facts.lmul)
             : llvm::StringRef();
  facts.storeIntrinsic =
      isLoad ? getRVVSelectedBodyStoreIntrinsic(facts.sew, facts.lmul)
             : llvm::StringRef();
  facts.stridedStoreIntrinsic =
      isStore ? getRVVSelectedBodyMaskedStridedStoreIntrinsic(facts.sew,
                                                              facts.lmul)
              : llvm::StringRef();
  facts.compareIntrinsic = getRVVSelectedBodyCompareIntrinsicForPredicate(
      facts.comparePredicateKind, facts.sew, facts.lmul);
  facts.computedMaskMemoryRouteFamilyPlanID =
      kRVVComputedMaskMemoryRouteFamilyPlanID;
  facts.computedMaskMemoryMaskProducerSource =
      getComputedMaskMemoryProducerSource(operation);
  facts.maskTailPolicyRouteFamilyPlanID = kRVVMaskTailPolicyRouteFamilyPlanID;
  facts.maskTailPolicyOwner = kRVVComputedMaskMemoryMaskTailPolicyOwner;
  facts.maskRole = kRVVMaskedPredicateMaskRole;
  facts.maskSource = kRVVMaskedCompareMaskSource;
  facts.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  facts.inactiveLaneContract =
      getComputedMaskMemoryInactiveLaneContract(operation);
  facts.maskedPassthroughLayout =
      getComputedMaskMemoryPassthroughLayout(operation);
  facts.maskedMemoryLayout = getComputedMaskMemoryLayout(operation);
  facts.stridedMemoryLayout = getComputedMaskMemoryLayout(operation);
  facts.sourceMemoryForm = getComputedMaskMemorySourceMemoryForm(operation);
  facts.destinationMemoryForm =
      getComputedMaskMemoryDestinationMemoryForm(operation);
  facts.sourceStrideSource =
      isLoad ? llvm::StringRef(kRVVSourceByteStrideSource) : llvm::StringRef();
  facts.destinationStrideSource =
      isStore ? llvm::StringRef(kRVVDestinationByteStrideSource)
              : llvm::StringRef();
  facts.sourceStrideCType = isLoad ? llvm::StringRef("size_t")
                                   : llvm::StringRef();
  facts.destinationStrideCType = isStore ? llvm::StringRef("size_t")
                                         : llvm::StringRef();
  facts.sourceStrideUnit = isLoad ? llvm::StringRef("byte") : llvm::StringRef();
  facts.destinationStrideUnit =
      isStore ? llvm::StringRef("byte") : llvm::StringRef();
  facts.runtimeABIParameters =
      getRVVComputedMaskStridedMemoryRuntimeABIParameters(operation);

  RVVRouteOperandBindingPlan plan =
      buildComputedMaskStridedMemoryRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    facts.logicalOperands.push_back(binding.logicalOperand);

  return facts;
}







std::optional<RVVPlainSegment2MemoryRouteFacts>
getRVVPlainSegment2MemoryRouteFacts(RVVSelectedBodyOperationKind operation) {
  const bool isDeinterleave =
      operation == RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
  const bool isInterleave =
      operation == RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
  if (!isDeinterleave && !isInterleave)
    return std::nullopt;

  RVVPlainSegment2MemoryRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = getSegment2MemoryRouteFamilyMemoryForm(operation);
  facts.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
  facts.lmul = tcrv::rvv::getRVVLMULM1();
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getSegment2MemoryRuntimeABIOrder(operation);
  facts.targetLeafProfile = getSegment2MemoryTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      getSegment2MemoryProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations = kRVVSegment2RequiredHeaderDeclarations;
  facts.cTypeMappingSummary = getSegment2MemoryCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID =
      getExpectedRVVRouteOperandBindingPlanID(operation);
  facts.typedComputeOpName =
      isDeinterleave ? llvm::StringRef("tcrv_rvv.move")
                     : llvm::StringRef("tcrv_rvv.segment2_store");
  facts.segment2MemoryRouteFamilyPlanID = kRVVSegment2MemoryRouteFamilyPlanID;
  facts.segment2Direction =
      isDeinterleave ? llvm::StringRef("deinterleave-load")
                     : llvm::StringRef("interleave-store");
  facts.usesDeinterleaveLoad = isDeinterleave;
  facts.usesInterleaveStore = isInterleave;
  facts.segmentMemoryLayout = getSegment2MemoryLayout(operation);
  facts.sourceMemoryForm =
      isDeinterleave ? llvm::StringRef(kRVVSegment2SourceMemoryForm)
                     : llvm::StringRef(kRVVUnitStrideSourceMemoryForm);
  facts.destinationMemoryForm =
      isInterleave ? llvm::StringRef(kRVVSegment2InterleavedDestinationMemoryForm)
                   : llvm::StringRef(kRVVDestinationMemoryForm);
  facts.segmentCount = 2;
  facts.segmentTupleCType =
      getRVVSelectedBodySegmentTupleCType(facts.sew, facts.lmul,
                                          facts.segmentCount);
  facts.segmentLoadIntrinsic =
      isDeinterleave ? getRVVSelectedBodySegmentLoadIntrinsic(
                           facts.sew, facts.lmul, facts.segmentCount)
                     : llvm::StringRef();
  facts.segmentStoreIntrinsic =
      isInterleave ? getRVVSelectedBodySegmentStoreIntrinsic(
                         facts.sew, facts.lmul, facts.segmentCount)
                   : llvm::StringRef();
  facts.segmentFieldExtractIntrinsic =
      isInterleave ? getRVVSelectedBodySegmentTupleCreateIntrinsic(
                         facts.sew, facts.lmul, facts.segmentCount)
                   : getRVVSelectedBodySegmentFieldExtractIntrinsic(
                         facts.sew, facts.lmul, facts.segmentCount);
  facts.field0Role =
      isInterleave ? llvm::StringRef(kRVVSegment2Field0InputRole)
                   : llvm::StringRef(kRVVSegment2Field0Role);
  facts.field1Role =
      isInterleave ? llvm::StringRef(kRVVSegment2Field1InputRole)
                   : llvm::StringRef(kRVVSegment2Field1Role);
  facts.field0Name = "field0_vec";
  facts.field1Name = "field1_vec";
  facts.field0SourceMemoryForm =
      isInterleave ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                   : llvm::StringRef();
  facts.field1SourceMemoryForm =
      isInterleave ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                   : llvm::StringRef();
  facts.field0DestinationMemoryForm =
      isDeinterleave ? llvm::StringRef(kRVVDestinationMemoryForm)
                     : llvm::StringRef();
  facts.field1DestinationMemoryForm =
      isDeinterleave ? llvm::StringRef(kRVVDestinationMemoryForm)
                     : llvm::StringRef();
  facts.runtimeABIParameters =
      getRVVPlainSegment2MemoryRuntimeABIParameters(operation);

  RVVRouteOperandBindingPlan plan =
      buildPlainSegment2MemoryRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    facts.logicalOperands.push_back(binding.logicalOperand);

  return facts;
}

std::optional<RVVComputedMaskSegment2MemoryRouteFacts>
getRVVComputedMaskSegment2MemoryRouteFacts(
    RVVSelectedBodyOperationKind operation) {
  const bool isLoad =
      operation == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
  const bool isRuntimeScalarLoad =
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskSegment2LoadUnitStore;
  const bool isStore =
      operation == RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
  const bool isRuntimeScalarStore =
      operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarComputedMaskSegment2StoreUnitLoad;
  const bool isUpdate =
      operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isStoreLike = isStore || isRuntimeScalarStore || isUpdate;
  const bool isLoadLike = isLoad || isRuntimeScalarLoad;
  if (!isLoadLike && !isStoreLike)
    return std::nullopt;

  RVVComputedMaskSegment2MemoryRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm = getComputedMaskMemoryRouteFamilyMemoryForm(operation);
  facts.sew = tcrv::rvv::getRVVFirstSliceSEWBits();
  facts.lmul = tcrv::rvv::getRVVLMULM1();
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.runtimeABIOrder = getComputedMaskMemoryRuntimeABIOrder(operation);
  facts.targetLeafProfile = getComputedMaskMemoryTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      getComputedMaskMemoryProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations =
      getComputedMaskMemoryRequiredHeaderDeclarations(operation);
  facts.cTypeMappingSummary =
      getComputedMaskMemoryCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID =
      getExpectedRVVRouteOperandBindingPlanID(operation);
  facts.typedComputeOpName =
      isLoadLike ? llvm::StringRef("tcrv_rvv.masked_segment2_load")
      : (isStore || isRuntimeScalarStore)
            ? llvm::StringRef("tcrv_rvv.masked_segment2_store")
            : llvm::StringRef("tcrv_rvv.binary");
  facts.comparePredicateKind = getComputedMaskMemoryComparePredicateKind(operation);
  facts.computedMaskMemoryRouteFamilyPlanID =
      kRVVComputedMaskMemoryRouteFamilyPlanID;
  facts.computedMaskMemoryMaskProducerSource =
      getComputedMaskMemoryProducerSource(operation);
  facts.maskTailPolicyRouteFamilyPlanID = kRVVMaskTailPolicyRouteFamilyPlanID;
  facts.maskTailPolicyOwner = kRVVComputedMaskMemoryMaskTailPolicyOwner;
  facts.maskRole = kRVVMaskedPredicateMaskRole;
  facts.maskSource = kRVVMaskedCompareMaskSource;
  facts.maskMemoryForm = kRVVComputedMaskMemoryMaskMemoryForm;
  facts.inactiveLaneContract =
      getComputedMaskMemoryInactiveLaneContract(operation);
  facts.maskedPassthroughLayout =
      getComputedMaskMemoryPassthroughLayout(operation);
  facts.segmentMemoryLayout = getComputedMaskMemoryLayout(operation);
  facts.sourceMemoryForm = getComputedMaskMemorySourceMemoryForm(operation);
  facts.destinationMemoryForm =
      getComputedMaskMemoryDestinationMemoryForm(operation);
  facts.segmentCount = 2;
  facts.segmentTupleCType =
      getRVVSelectedBodySegmentTupleCType(facts.sew, facts.lmul,
                                          facts.segmentCount);
  facts.segmentLoadIntrinsic =
      isLoadLike ? getRVVSelectedBodyMaskedSegmentLoadIntrinsic(
                       facts.sew, facts.lmul, facts.segmentCount)
                 : llvm::StringRef();
  facts.segmentStoreIntrinsic =
      isLoadLike ? getRVVSelectedBodySegmentTupleCreateIntrinsic(
                       facts.sew, facts.lmul, facts.segmentCount)
      : isStoreLike ? getRVVSelectedBodyMaskedSegmentStoreIntrinsic(
                          facts.sew, facts.lmul, facts.segmentCount)
                    : llvm::StringRef();
  facts.segmentFieldExtractIntrinsic =
      isLoadLike ? getRVVSelectedBodySegmentFieldExtractIntrinsic(
                       facts.sew, facts.lmul, facts.segmentCount)
                 : getRVVSelectedBodySegmentTupleCreateIntrinsic(
                       facts.sew, facts.lmul, facts.segmentCount);
  facts.rhsScalarSplatIntrinsic =
      (isRuntimeScalarLoad || isRuntimeScalarStore)
          ? getRVVSelectedBodyScalarSplatIntrinsic(facts.sew, facts.lmul)
          : llvm::StringRef();
  facts.segment2UpdateArithmeticKind = isUpdate ? llvm::StringRef("add")
                                                : llvm::StringRef();
  facts.segment2UpdateArithmeticIntrinsic =
      isUpdate ? getRVVSelectedBodyArithmeticIntrinsic(operation, facts.sew,
                                                       facts.lmul)
               : llvm::StringRef();
  facts.field0Role =
      isLoadLike ? llvm::StringRef(kRVVSegment2Field0Role)
                 : llvm::StringRef(kRVVSegment2Field0InputRole);
  facts.field1Role =
      isLoadLike ? llvm::StringRef(kRVVSegment2Field1Role)
                 : llvm::StringRef(kRVVSegment2Field1InputRole);
  facts.field0Name =
      isLoadLike ? llvm::StringRef("masked_segment2_field0_vec")
      : isUpdate ? llvm::StringRef("masked_segment2_update_field0_vec")
                 : llvm::StringRef("masked_segment2_store_field0_vec");
  facts.field1Name =
      isLoadLike ? llvm::StringRef("masked_segment2_field1_vec")
      : isUpdate ? llvm::StringRef("masked_segment2_update_field1_vec")
                 : llvm::StringRef("masked_segment2_store_field1_vec");
  facts.field0SourceMemoryForm =
      isStoreLike ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                  : llvm::StringRef();
  facts.field1SourceMemoryForm =
      isStoreLike ? llvm::StringRef(kRVVUnitStrideSourceMemoryForm)
                  : llvm::StringRef();
  facts.field0DestinationMemoryForm =
      isLoadLike ? llvm::StringRef(kRVVDestinationMemoryForm)
                 : llvm::StringRef();
  facts.field1DestinationMemoryForm =
      isLoadLike ? llvm::StringRef(kRVVDestinationMemoryForm)
                 : llvm::StringRef();
  facts.runtimeABIParameters =
      getRVVComputedMaskSegment2MemoryRuntimeABIParameters(operation);

  RVVRouteOperandBindingPlan plan =
      buildComputedMaskSegment2MemoryRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary =
      stringifyRVVRouteOperandBindingPlan(plan);
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    facts.logicalOperands.push_back(binding.logicalOperand);

  return facts;
}













llvm::StringRef getRVVSelectedBodyStandaloneReductionInactiveNeutralLiteral(
    RVVSelectedBodyOperationKind operation, std::int64_t sew) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
    return "0";
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
    return sew == 64 ? llvm::StringRef("9223372036854775807")
                     : llvm::StringRef("2147483647");
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return sew == 64 ? llvm::StringRef("(-9223372036854775807-1)")
                     : llvm::StringRef("(-2147483647-1)");
  default:
    return {};
  }
}

bool isRVVCompareSelectRouteFactsOperation(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::CmpSelect:
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
  case RVVSelectedBodyOperationKind::F32ClampSelect:
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return true;
  default:
    return false;
  }
}

llvm::StringRef getDefaultRVVCompareSelectRouteFactsPredicate(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::CmpSelect:
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "slt";
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
    return "sle";
  case RVVSelectedBodyOperationKind::F32ClampSelect:
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return "slt";
  default:
    return {};
  }
}

bool isSupportedRVVCompareSelectRouteFactsPredicate(
    RVVSelectedBodyOperationKind operation, llvm::StringRef predicate,
    llvm::StringRef secondaryPredicate) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::CmpSelect:
    return secondaryPredicate.empty() &&
           isSupportedPlainCompareSelectPredicateKind(predicate);
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return secondaryPredicate.empty() &&
           (predicate == "slt" || predicate == "sle");
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return secondaryPredicate.empty() && predicate == "sle";
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
    return predicate == "sle" && secondaryPredicate == "sle";
  case RVVSelectedBodyOperationKind::F32ClampSelect:
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return predicate == "slt" && secondaryPredicate == "slt";
  default:
    return false;
  }
}

std::optional<RVVCompareSelectRouteFacts>
getRVVCompareSelectRouteFacts(RVVSelectedBodyOperationKind operation) {
  llvm::StringRef predicate =
      getDefaultRVVCompareSelectRouteFactsPredicate(operation);
  llvm::StringRef secondaryPredicate;
  if (operation == RVVSelectedBodyOperationKind::
                       RuntimeScalarDualCompareMaskAndSelect)
    secondaryPredicate = "sle";
  if (operation == RVVSelectedBodyOperationKind::F32ClampSelect)
    secondaryPredicate = "slt";
  if (operation == RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    secondaryPredicate = "slt";
  if (predicate.empty())
    return std::nullopt;
  return getRVVCompareSelectRouteFacts(
      operation, tcrv::rvv::getRVVFirstSliceSEWBits(),
      tcrv::rvv::getRVVLMULM1(), predicate, secondaryPredicate);
}

std::optional<RVVCompareSelectRouteFacts>
getRVVCompareSelectRouteFacts(RVVSelectedBodyOperationKind operation,
                              std::int64_t sew, llvm::StringRef lmul,
                              llvm::StringRef comparePredicateKind,
                              llvm::StringRef secondaryComparePredicateKind) {
  if (!isRVVCompareSelectRouteFactsOperation(operation))
    return std::nullopt;
  if (!isSupportedRVVCompareSelectRouteFactsPredicate(
          operation, comparePredicateKind, secondaryComparePredicateKind))
    return std::nullopt;
  if (operation != RVVSelectedBodyOperationKind::CmpSelect &&
      !isSupportedTypedComputedMaskSelectRouteConfig(sew, lmul))
    return std::nullopt;

  if (operation == RVVSelectedBodyOperationKind::F32ClampSelect ||
      operation == RVVSelectedBodyOperationKind::DequantClampF32Epilogue) {
    const bool isDequantClamp =
        operation == RVVSelectedBodyOperationKind::DequantClampF32Epilogue;
    if (sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
        lmul != tcrv::rvv::getRVVLMULM1())
      return std::nullopt;

    const llvm::StringRef elementTypeName =
        getRVVSelectedBodyFloatElementTypeName(sew);
    const llvm::StringRef constPointerCType =
        isDequantClamp ? getRVVSelectedBodyConstInputPointerCType(sew)
                       : getRVVSelectedBodyFloatConstInputPointerCType(sew);
    const llvm::StringRef outputPointerCType =
        getRVVSelectedBodyFloatOutputPointerCType(sew);
    const llvm::StringRef vectorTypeName =
        getRVVSelectedBodyFloatVectorTypeName(sew, lmul);
    const llvm::StringRef vectorCType =
        getRVVSelectedBodyFloatVectorCType(sew, lmul);
    const llvm::StringRef maskTypeName =
        getRVVSelectedBodyFloatMaskTypeName(sew, lmul);
    const llvm::StringRef maskCType = getRVVSelectedBodyMaskCType(sew, lmul);
    const llvm::StringRef setVLIntrinsic =
        getRVVSelectedBodySetVLIntrinsic(sew, lmul);
    const llvm::StringRef vectorLoadIntrinsic =
        getRVVSelectedBodyFloatVectorLoadIntrinsic(sew, lmul);
    const llvm::StringRef scalarSplatIntrinsic =
        getRVVSelectedBodyFloatScalarSplatIntrinsic(sew, lmul);
    const llvm::StringRef compareIntrinsic =
        getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
            comparePredicateKind, sew, lmul);
    const llvm::StringRef secondaryCompareIntrinsic =
        getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
            secondaryComparePredicateKind, sew, lmul);
    const llvm::StringRef selectIntrinsic =
        getRVVSelectedBodyFloatSelectIntrinsic(sew, lmul);
    const llvm::StringRef storeIntrinsic =
        getRVVSelectedBodyFloatStoreIntrinsic(sew, lmul);
    if (elementTypeName.empty() || constPointerCType.empty() ||
        outputPointerCType.empty() || vectorTypeName.empty() ||
        vectorCType.empty() || maskTypeName.empty() || maskCType.empty() ||
        setVLIntrinsic.empty() || vectorLoadIntrinsic.empty() ||
        scalarSplatIntrinsic.empty() || compareIntrinsic.empty() ||
        secondaryCompareIntrinsic.empty() || selectIntrinsic.empty() ||
        storeIntrinsic.empty())
      return std::nullopt;

    RVVCompareSelectRouteFacts facts;
    facts.operation = operation;
    facts.memoryForm =
        isDequantClamp
            ? RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue
            : RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect;
    facts.elementTypeName = elementTypeName;
    facts.sew = sew;
    facts.lmul = lmul;
    facts.tailPolicy = "agnostic";
    facts.maskPolicy = "agnostic";
    facts.runtimeABIOrder =
        isDequantClamp ? llvm::StringRef(kRVVDequantClampF32EpilogueRuntimeABIOrder)
                       : llvm::StringRef(kRVVF32ClampSelectRuntimeABIOrder);
    facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
    facts.targetLeafProfile =
        isDequantClamp
            ? llvm::StringRef(kRVVDequantClampF32EpilogueTargetLeafProfile)
            : llvm::StringRef(kRVVF32ClampSelectTargetLeafProfile);
    facts.providerSupportedMirror =
        isDequantClamp
            ? llvm::StringRef(
                  kRVVDequantClampF32EpilogueProviderSupportedMirror)
            : llvm::StringRef(kRVVF32ClampSelectProviderSupportedMirror);
    facts.requiredHeaderDeclarations =
        isDequantClamp
            ? llvm::StringRef(
                  kRVVDequantClampF32EpilogueRequiredHeaderDeclarations)
            : llvm::StringRef(kRVVF32ClampSelectRequiredHeaderDeclarations);
    facts.cTypeMappingSummary =
        isDequantClamp
            ? llvm::StringRef(kRVVDequantClampF32EpilogueCTypeMappingSummary)
            : llvm::StringRef(kRVVF32ClampSelectCTypeMappingSummary);
    facts.routeOperandBindingPlanID =
        getExpectedRVVRouteOperandBindingPlanID(operation);
    facts.typedComputeOpName = "tcrv_rvv.select";
    facts.vlCType = "size_t";
    facts.vectorTypeName = vectorTypeName;
    facts.vectorCType = vectorCType;
    facts.maskTypeName = maskTypeName;
    facts.maskCType = maskCType;
    facts.setVLIntrinsic = setVLIntrinsic;
    facts.vectorLoadIntrinsic = vectorLoadIntrinsic;
    facts.comparePredicateKind = comparePredicateKind;
    facts.secondaryComparePredicateKind = secondaryComparePredicateKind;
    facts.rhsScalarSplatIntrinsic = scalarSplatIntrinsic;
    facts.compareIntrinsic = compareIntrinsic;
    facts.secondaryCompareIntrinsic = secondaryCompareIntrinsic;
    facts.selectIntrinsic = selectIntrinsic;
    facts.storeIntrinsic = storeIntrinsic;
    const RVVSelectedBodyOperationProfile &operationProfile =
        getRVVSelectedBodyOperationProfile(operation);
    facts.resultName = operationProfile.resultName;
    facts.maskName = operationProfile.maskName;
    facts.computedMaskSelectRouteFamilyPlanID =
        isDequantClamp
            ? llvm::StringRef(kRVVDequantClampF32EpilogueRouteFamilyPlanID)
            : llvm::StringRef(kRVVF32ClampSelectRouteFamilyPlanID);
    facts.computedMaskSelectMaskProducerSource =
        isDequantClamp
            ? llvm::StringRef(
                  "i32-to-f32-dequant-then-two-compare-two-select-f32-clamp-same-vl-scope")
            : llvm::StringRef("two-compare-two-select-f32-clamp-same-vl-scope");
    facts.maskTailPolicyRouteFamilyPlanID =
        kRVVMaskTailPolicyRouteFamilyPlanID;
    facts.maskTailPolicyOwner = kRVVComputedMaskSelectMaskTailPolicyOwner;
    facts.maskRole =
        isDequantClamp
            ? llvm::StringRef(
                  "predicate-mask-produced-by-dequantized-f32-bound-compare")
            : llvm::StringRef("predicate-mask-produced-by-f32-bound-compare");
    facts.maskSource =
        isDequantClamp
            ? llvm::StringRef(
                  "dequantized-f32-lower-upper-bound-compare-produced-masks")
            : llvm::StringRef("f32-lower-upper-bound-compare-produced-masks");
    facts.maskMemoryForm =
        isDequantClamp ? llvm::StringRef("two-dequantized-f32-compare-produced-masks")
                       : llvm::StringRef("two-f32-compare-produced-masks");
    facts.selectLayout = "clamp-lower-then-upper";
    facts.trueValueRole = "lower-then-upper-bound-vector-when-mask-true";
    facts.falseValueRole =
        isDequantClamp ? llvm::StringRef(
                             "dequantized-or-lower-clamped-vector-when-mask-false")
                       : llvm::StringRef(
                             "input-or-lower-clamped-vector-when-mask-false");
    facts.selectedResultRole = "f32-clamped-vector-output";
    facts.lowerBoundRole = "lower-bound-scalar-value";
    facts.upperBoundRole = "upper-bound-scalar-value";
    facts.lowerBoundCType = "float";
    facts.upperBoundCType = "float";
    facts.boundOrder = "lower-bound-before-upper-bound";
    facts.clampRelation =
        isDequantClamp
            ? llvm::StringRef(
                  "i32-input-runtime-scale-dequant-lower-select-then-upper-select-f32-runtime-bounds")
            : llvm::StringRef("input-lower-select-then-upper-select-f32-runtime-bounds");
    facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
    facts.destinationMemoryForm = kRVVDestinationMemoryForm;
    facts.indexedMemoryLayout =
        isDequantClamp
            ? llvm::StringRef(
                  "unit-stride-i32-input-runtime-scale-dequant-runtime-lower-upper-select-f32-output-runtime-abi")
            : llvm::StringRef(
                  "unit-stride-f32-input-runtime-lower-upper-select-output-runtime-abi");

    using support::RuntimeABIParameter;
    using support::RuntimeABIParameterOwnership;
    using support::RuntimeABIParameterRole;
    facts.runtimeABIParameters.push_back(RuntimeABIParameter(
        isDequantClamp ? "lhs" : "input", constPointerCType,
        RuntimeABIParameterRole::LHSInputBuffer,
        RuntimeABIParameterOwnership::TargetExportABIOwned));
    if (isDequantClamp)
      facts.runtimeABIParameters.push_back(RuntimeABIParameter(
          "scale", "float", RuntimeABIParameterRole::DequantScaleValue,
          RuntimeABIParameterOwnership::TargetExportABIOwned));
    facts.runtimeABIParameters.push_back(RuntimeABIParameter(
        "lower_bound", "float",
        RuntimeABIParameterRole::LowerBoundScalarValue,
        RuntimeABIParameterOwnership::TargetExportABIOwned));
    facts.runtimeABIParameters.push_back(RuntimeABIParameter(
        "upper_bound", "float",
        RuntimeABIParameterRole::UpperBoundScalarValue,
        RuntimeABIParameterOwnership::TargetExportABIOwned));
    facts.runtimeABIParameters.push_back(RuntimeABIParameter(
        "out", outputPointerCType, RuntimeABIParameterRole::OutputBuffer,
        RuntimeABIParameterOwnership::TargetExportABIOwned));
    facts.runtimeABIParameters.push_back(RuntimeABIParameter(
        "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
        RuntimeABIParameterOwnership::TargetExportABIOwned));

    RVVRouteOperandBindingPlan plan =
        buildCompareSelectRouteOperandBindingPlanFromFacts(facts);
    facts.routeOperandBindingSummary = stringifyRVVRouteOperandBindingPlan(plan);
    return facts;
  }

  const llvm::StringRef elementTypeName =
      getRVVSelectedBodyIntegerElementTypeName(sew);
  const llvm::StringRef scalarCType = getRVVSelectedBodySignedScalarCType(sew);
  const llvm::StringRef constPointerCType =
      getRVVSelectedBodyConstInputPointerCType(sew);
  const llvm::StringRef outputPointerCType =
      getRVVSelectedBodyOutputPointerCType(sew);
  const llvm::StringRef vectorTypeName =
      getRVVSelectedBodyVectorTypeName(sew, lmul);
  const llvm::StringRef vectorCType =
      getRVVSelectedBodySignedVectorCType(sew, lmul);
  const llvm::StringRef maskTypeName =
      getRVVSelectedBodyMaskTypeName(sew, lmul);
  const llvm::StringRef maskCType = getRVVSelectedBodyMaskCType(sew, lmul);
  const llvm::StringRef setVLIntrinsic =
      getRVVSelectedBodySetVLIntrinsic(sew, lmul);
  const llvm::StringRef vectorLoadIntrinsic =
      getRVVSelectedBodyVectorLoadIntrinsic(sew, lmul);
  const llvm::StringRef rhsScalarSplatIntrinsic =
      usesRuntimeScalarComputedMaskSelectProducer(operation)
          ? getRVVSelectedBodyScalarSplatIntrinsic(sew, lmul)
          : llvm::StringRef();
  const llvm::StringRef compareIntrinsic =
      getRVVSelectedBodyCompareIntrinsicForPredicate(comparePredicateKind, sew,
                                                     lmul);
  const llvm::StringRef secondaryCompareIntrinsic =
      usesDualCompareMaskAndSelect(operation) ? compareIntrinsic
                                              : llvm::StringRef();
  const llvm::StringRef maskAndIntrinsic =
      usesDualCompareMaskAndSelect(operation)
          ? getRVVSelectedBodyMaskAndIntrinsic(sew, lmul)
          : llvm::StringRef();
  const llvm::StringRef selectIntrinsic =
      getRVVSelectedBodySelectIntrinsic(sew, lmul);
  const llvm::StringRef storeIntrinsic =
      getRVVSelectedBodyStoreIntrinsic(sew, lmul);
  if (elementTypeName.empty() || scalarCType.empty() ||
      constPointerCType.empty() || outputPointerCType.empty() ||
      vectorTypeName.empty() || vectorCType.empty() || maskTypeName.empty() ||
      maskCType.empty() || setVLIntrinsic.empty() ||
      vectorLoadIntrinsic.empty() || compareIntrinsic.empty() ||
      selectIntrinsic.empty() || storeIntrinsic.empty())
    return std::nullopt;
  if (usesRuntimeScalarComputedMaskSelectProducer(operation) &&
      rhsScalarSplatIntrinsic.empty())
    return std::nullopt;
  if (usesDualCompareMaskAndSelect(operation) &&
      (secondaryCompareIntrinsic.empty() || maskAndIntrinsic.empty()))
    return std::nullopt;

  const bool isPlain = operation == RVVSelectedBodyOperationKind::CmpSelect;
  const bool isDual = usesDualCompareMaskAndSelect(operation);
  const bool isVectorComputed =
      operation == RVVSelectedBodyOperationKind::ComputedMaskSelect;
  const bool isRuntimeScalar =
      operation == RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect;

  RVVCompareSelectRouteFacts facts;
  facts.operation = operation;
  facts.memoryForm =
      isPlain ? RVVSelectedBodyMemoryForm::VectorRHSLoad
      : isVectorComputed ? RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect
      : isDual
          ? RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect
          : RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect;
  facts.elementTypeName = elementTypeName;
  facts.sew = sew;
  facts.lmul = lmul;
  facts.tailPolicy = "agnostic";
  facts.maskPolicy = "agnostic";
  facts.runtimeABIOrder =
      isPlain ? llvm::StringRef(kRVVGenericBinaryRuntimeABIOrder)
              : getComputedMaskSelectRuntimeABIOrder(operation);
  facts.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  facts.targetLeafProfile =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectTargetLeafProfile)
              : getComputedMaskSelectTargetLeafProfile(operation);
  facts.providerSupportedMirror =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectProviderSupportedMirror)
              : getComputedMaskSelectProviderSupportedMirror(operation);
  facts.requiredHeaderDeclarations =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectRequiredHeaderDeclarations)
              : getComputedMaskSelectHeaderDeclarations(operation);
  facts.cTypeMappingSummary =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectCTypeMappingSummary)
              : getComputedMaskSelectCTypeMappingSummary(operation);
  facts.routeOperandBindingPlanID =
      getExpectedRVVRouteOperandBindingPlanID(operation);
  facts.typedComputeOpName = "tcrv_rvv.select";
  facts.vlCType = "size_t";
  facts.vectorTypeName = vectorTypeName;
  facts.vectorCType = vectorCType;
  facts.maskTypeName = maskTypeName;
  facts.maskCType = maskCType;
  facts.setVLIntrinsic = setVLIntrinsic;
  facts.vectorLoadIntrinsic = vectorLoadIntrinsic;
  facts.comparePredicateKind = comparePredicateKind;
  facts.secondaryComparePredicateKind = secondaryComparePredicateKind;
  facts.rhsScalarSplatIntrinsic = rhsScalarSplatIntrinsic;
  facts.compareIntrinsic = compareIntrinsic;
  facts.secondaryCompareIntrinsic = secondaryCompareIntrinsic;
  facts.maskAndIntrinsic = maskAndIntrinsic;
  facts.selectIntrinsic = selectIntrinsic;
  facts.storeIntrinsic = storeIntrinsic;
  const RVVSelectedBodyOperationProfile &operationProfile =
      getRVVSelectedBodyOperationProfile(operation);
  facts.resultName = operationProfile.resultName;
  facts.maskName = operationProfile.maskName;
  facts.plainCompareSelectRouteFamilyPlanID =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectRouteFamilyPlanID)
              : llvm::StringRef();
  facts.computedMaskSelectRouteFamilyPlanID =
      isPlain ? llvm::StringRef()
              : llvm::StringRef(kRVVComputedMaskSelectRouteFamilyPlanID);
  facts.computedMaskSelectMaskProducerSource =
      isPlain ? llvm::StringRef()
              : getComputedMaskSelectMaskProducerSource(operation);
  facts.maskTailPolicyRouteFamilyPlanID =
      isPlain ? llvm::StringRef()
              : llvm::StringRef(kRVVMaskTailPolicyRouteFamilyPlanID);
  facts.maskTailPolicyOwner =
      isPlain ? llvm::StringRef()
              : llvm::StringRef(kRVVComputedMaskSelectMaskTailPolicyOwner);
  facts.maskRole = isPlain ? llvm::StringRef(kRVVMaskedPredicateMaskRole)
                           : getComputedMaskSelectMaskRole(operation);
  facts.maskSource = isPlain ? llvm::StringRef(kRVVMaskedCompareMaskSource)
                             : getComputedMaskSelectMaskSource(operation);
  facts.maskMemoryForm =
      isPlain ? llvm::StringRef(kRVVComputedMaskMemoryMaskMemoryForm)
              : getComputedMaskSelectMaskMemoryForm(operation);
  facts.maskComposition = isDual ? llvm::StringRef("and") : llvm::StringRef();
  facts.inactiveLaneContract =
      isPlain ? llvm::StringRef(kRVVMaskedInactiveLaneContract)
              : llvm::StringRef();
  facts.maskedPassthroughLayout =
      isPlain ? llvm::StringRef(kRVVMaskedPassthroughLayout)
              : llvm::StringRef();
  facts.selectLayout =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectLayout)
              : llvm::StringRef("select-true-value-when-mask-else-false-value");
  facts.trueValueRole =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectTrueValueRole)
              : llvm::StringRef("true-value-vector-when-mask-true");
  facts.falseValueRole =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectFalseValueRole)
              : llvm::StringRef("false-value-vector-when-mask-false");
  facts.selectedResultRole =
      isPlain ? llvm::StringRef(kRVVPlainCompareSelectSelectedResultRole)
              : llvm::StringRef("selected-vector-output");
  facts.runtimeScalarThresholdRole =
      isRuntimeScalar ? llvm::StringRef("rhs-scalar-value")
      : isDual ? llvm::StringRef("rhs-scalar-value,rhs-secondary-scalar-value")
               : llvm::StringRef();
  facts.runtimeScalarThresholdCType =
      (isRuntimeScalar || isDual) ? scalarCType : llvm::StringRef();
  facts.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
  facts.destinationMemoryForm = kRVVDestinationMemoryForm;
  facts.indexedMemoryLayout =
      isVectorComputed ? llvm::StringRef(kRVVComputedMaskSelectMemoryLayout)
      : isRuntimeScalar
          ? llvm::StringRef(
                "unit-stride-lhs-runtime-scalar-threshold-true-false-select-output-runtime-abi")
          : llvm::StringRef();

  using support::RuntimeABIParameter;
  using support::RuntimeABIParameterOwnership;
  using support::RuntimeABIParameterRole;
  auto addParameter = [&](llvm::StringRef cName, llvm::StringRef cType,
                          RuntimeABIParameterRole role) {
    facts.runtimeABIParameters.push_back(RuntimeABIParameter(
        cName, cType, role, RuntimeABIParameterOwnership::TargetExportABIOwned));
  };
  if (isPlain) {
    addParameter("lhs", constPointerCType,
                 RuntimeABIParameterRole::LHSInputBuffer);
    addParameter("rhs", constPointerCType,
                 RuntimeABIParameterRole::RHSInputBuffer);
    addParameter("out", outputPointerCType,
                 RuntimeABIParameterRole::OutputBuffer);
    addParameter("n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount);
  } else if (isVectorComputed) {
    addParameter("cmp_lhs", constPointerCType,
                 RuntimeABIParameterRole::LHSInputBuffer);
    addParameter("cmp_rhs", constPointerCType,
                 RuntimeABIParameterRole::RHSInputBuffer);
    addParameter("true_value", constPointerCType,
                 RuntimeABIParameterRole::TrueValueInputBuffer);
    addParameter("false_value", constPointerCType,
                 RuntimeABIParameterRole::FalseValueInputBuffer);
    addParameter("out", outputPointerCType,
                 RuntimeABIParameterRole::OutputBuffer);
    addParameter("n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount);
  } else if (isRuntimeScalar) {
    addParameter("lhs", constPointerCType,
                 RuntimeABIParameterRole::LHSInputBuffer);
    addParameter("rhs_scalar", scalarCType,
                 RuntimeABIParameterRole::RHSScalarValue);
    addParameter("true_value", constPointerCType,
                 RuntimeABIParameterRole::TrueValueInputBuffer);
    addParameter("false_value", constPointerCType,
                 RuntimeABIParameterRole::FalseValueInputBuffer);
    addParameter("out", outputPointerCType,
                 RuntimeABIParameterRole::OutputBuffer);
    addParameter("n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount);
  } else {
    addParameter("cmp_lhs_a", constPointerCType,
                 RuntimeABIParameterRole::LHSInputBuffer);
    addParameter("rhs_scalar_a", scalarCType,
                 RuntimeABIParameterRole::RHSScalarValue);
    addParameter("cmp_lhs_b", constPointerCType,
                 RuntimeABIParameterRole::RHSInputBuffer);
    addParameter("rhs_scalar_b", scalarCType,
                 RuntimeABIParameterRole::RHSSecondaryScalarValue);
    addParameter("true_value", constPointerCType,
                 RuntimeABIParameterRole::TrueValueInputBuffer);
    addParameter("false_value", constPointerCType,
                 RuntimeABIParameterRole::FalseValueInputBuffer);
    addParameter("out", outputPointerCType,
                 RuntimeABIParameterRole::OutputBuffer);
    addParameter("n", "size_t",
                 RuntimeABIParameterRole::RuntimeElementCount);
  }

  RVVRouteOperandBindingPlan plan =
      buildCompareSelectRouteOperandBindingPlanFromFacts(facts);
  facts.routeOperandBindingSummary = stringifyRVVRouteOperandBindingPlan(plan);

  return facts;
}

std::optional<RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts>
getRVVRuntimeScalarDualCompareMaskAndSelectRouteFacts(
    RVVSelectedBodyOperationKind operation) {
  return getRVVRuntimeScalarDualCompareMaskAndSelectRouteFacts(
      operation, tcrv::rvv::getRVVFirstSliceSEWBits(),
      tcrv::rvv::getRVVLMULM1());
}

std::optional<RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts>
getRVVRuntimeScalarDualCompareMaskAndSelectRouteFacts(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul) {
  return getRVVCompareSelectRouteFacts(operation, sew, lmul, "sle", "sle");
}












bool isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::RuntimeScalarSplatStore;
}

bool isRVVSelectedBodyPlainCompareSelectRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyPlainCompareSelectRouteOperation(operation);
}

bool isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyWideningConversionRouteOperation(operation);
}

bool isRVVSelectedBodyDequantizationRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyDequantizationRouteOperation(operation);
}

bool isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect:
  case RVVSelectedBodyOperationKind::F32ClampSelect:
  case RVVSelectedBodyOperationKind::DequantClampF32Epilogue:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyPlainStandaloneReductionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyComputedMaskStandaloneReductionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVSelectedBodyStandaloneReductionRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyPlainStandaloneReductionRouteFamilyConsumer(
             operation) ||
         isRVVSelectedBodyComputedMaskStandaloneReductionRouteFamilyConsumer(
             operation);
}

namespace {

} // namespace

void addRVVSelectedBodySegment2MemoryRouteFamilyMetadataMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::SmallVectorImpl<support::ArtifactMetadataEntry> &metadata) {
  if (description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskSegment2LoadUnitStore) {
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.segment_memory_layout",
                        description.segmentMemoryLayout});
    metadata.push_back({"tcrv_rvv.segment_count",
                        llvm::Twine(description.segmentCount).str()});
    metadata.push_back({"tcrv_rvv.segment_tuple_c_type",
                        description.segmentTupleCType});
    metadata.push_back({"tcrv_rvv.segment_load_intrinsic",
                        description.segmentLoadIntrinsic});
    metadata.push_back({"tcrv_rvv.segment_tuple_create_intrinsic",
                        description.segmentStoreIntrinsic});
    metadata.push_back({"tcrv_rvv.segment_field_extract_intrinsic",
                        description.segmentFieldExtractIntrinsic});
    metadata.push_back({"tcrv_rvv.field0_role", description.field0Role});
    metadata.push_back({"tcrv_rvv.field1_role", description.field1Role});
    metadata.push_back({"tcrv_rvv.field0_name", description.field0Name});
    metadata.push_back({"tcrv_rvv.field1_name", description.field1Name});
    metadata.push_back({"tcrv_rvv.field0_destination_memory_form",
                        description.field0DestinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field1_destination_memory_form",
                        description.field1DestinationMemoryForm});
    return;
  }
	  if (description.operation ==
	          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
	      description.operation ==
	          RVVSelectedBodyOperationKind::
	              RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
	      description.operation ==
	          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad) {
    const bool isUpdate =
        description.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
    metadata.push_back({"tcrv_rvv.segment_memory_layout",
                        description.segmentMemoryLayout});
    metadata.push_back({"tcrv_rvv.segment_count",
                        llvm::Twine(description.segmentCount).str()});
    metadata.push_back({"tcrv_rvv.segment_tuple_c_type",
                        description.segmentTupleCType});
    metadata.push_back({"tcrv_rvv.segment_store_intrinsic",
                        description.segmentStoreIntrinsic});
    metadata.push_back({"tcrv_rvv.segment_tuple_create_intrinsic",
                        description.segmentFieldExtractIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field0_role", description.field0Role});
    metadata.push_back({"tcrv_rvv.field1_role", description.field1Role});
    metadata.push_back({"tcrv_rvv.field0_name", description.field0Name});
    metadata.push_back({"tcrv_rvv.field1_name", description.field1Name});
    metadata.push_back({"tcrv_rvv.field0_source_memory_form",
                        description.field0SourceMemoryForm});
    metadata.push_back({"tcrv_rvv.field1_source_memory_form",
                        description.field1SourceMemoryForm});
    if (isUpdate) {
      metadata.push_back({"tcrv_rvv.segment2_update_arithmetic_kind",
                          description.segment2UpdateArithmeticKind});
      metadata.push_back({"tcrv_rvv.segment2_update_arithmetic_intrinsic",
                          description.segment2UpdateArithmeticIntrinsic});
    }
    return;
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore) {
    metadata.push_back({"tcrv_rvv.segment_memory_layout",
                        description.segmentMemoryLayout});
    metadata.push_back({"tcrv_rvv.segment_count",
                        llvm::Twine(description.segmentCount).str()});
    metadata.push_back({"tcrv_rvv.segment_tuple_c_type",
                        description.segmentTupleCType});
    metadata.push_back({"tcrv_rvv.segment_load_intrinsic",
                        description.segmentLoadIntrinsic});
    metadata.push_back({"tcrv_rvv.segment_field_extract_intrinsic",
                        description.segmentFieldExtractIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field0_role", description.field0Role});
    metadata.push_back({"tcrv_rvv.field1_role", description.field1Role});
    metadata.push_back({"tcrv_rvv.field0_name", description.field0Name});
    metadata.push_back({"tcrv_rvv.field1_name", description.field1Name});
    metadata.push_back({"tcrv_rvv.field0_destination_memory_form",
                        description.field0DestinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field1_destination_memory_form",
                        description.field1DestinationMemoryForm});
    return;
  }
  if (description.operation ==
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad) {
    metadata.push_back({"tcrv_rvv.segment_memory_layout",
                        description.segmentMemoryLayout});
    metadata.push_back({"tcrv_rvv.segment_count",
                        llvm::Twine(description.segmentCount).str()});
    metadata.push_back({"tcrv_rvv.segment_tuple_c_type",
                        description.segmentTupleCType});
    metadata.push_back({"tcrv_rvv.segment_store_intrinsic",
                        description.segmentStoreIntrinsic});
    metadata.push_back({"tcrv_rvv.segment_tuple_create_intrinsic",
                        description.segmentFieldExtractIntrinsic});
    metadata.push_back(
        {"tcrv_rvv.source_memory_form", description.sourceMemoryForm});
    metadata.push_back({"tcrv_rvv.destination_memory_form",
                        description.destinationMemoryForm});
    metadata.push_back({"tcrv_rvv.field0_role", description.field0Role});
    metadata.push_back({"tcrv_rvv.field1_role", description.field1Role});
    metadata.push_back({"tcrv_rvv.field0_name", description.field0Name});
    metadata.push_back({"tcrv_rvv.field1_name", description.field1Name});
    metadata.push_back({"tcrv_rvv.field0_source_memory_form",
                        description.field0SourceMemoryForm});
    metadata.push_back({"tcrv_rvv.field1_source_memory_form",
                        description.field1SourceMemoryForm});
  }
}

llvm::Expected<RVVSelectedBodyRouteAnalysis>
analyzeRVVSelectedBodyRoute(const VariantEmitCLowerableRequest &request) {
  if (!request.getVariant())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return std::move(error);
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return std::move(error);
  std::optional<RVVPreRealizedSelectedBodyMatch> preRealizedBody =
      findFirstPreRealizedRVVSelectedBodyMatch(request.getVariant());
  if (preRealizedBody)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("RVV selected-body realization boundary must run before "
                    "route facts are collected for selected variant @") +
        request.getVariant().getSymName() +
        "; route planning/provider saw a pre-realized tcrv_rvv body '" +
        preRealizedBody->bodyOp->getName().getStringRef() +
        "' owned by selected-body realization owner '" +
        preRealizedBody->familyName +
        "' instead of a realized setvl/with_vl body");

  llvm::Expected<RVVSelectedBodyRouteSlice> slice =
      collectRVVSelectedBodyRouteSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  tcrv::rvv::RVVCompileTimeConfig config =
      tcrv::rvv::getRVVSetVLCompileTimeConfig(slice->setvl);
  if (llvm::Error error =
          validateRVVSelectedBodyTypedConfigFacts(*slice, config))
    return std::move(error);
  // The deferred-wide (N3) product-reduce-dequant realization carries its SOURCE
  // strip config (sew8/m2) on the setvl, but the route's LOGICAL profile is the
  // i32m1/f32m1 RESULT config -- identical to the narrow dequant route (whose
  // setvl coincidentally carries that result config). The typed config facts were
  // already validated against the structural source/product/accumulate/result
  // types above; here the route description/configContract are normalized to the
  // result config so the dequant route profile derivation, c-type mapping, and
  // header mirror match the narrow path (I5: the result type is structural in the
  // reduce/dequant ops, not inferred from a name).
  if (slice->arithmeticKind ==
      RVVSelectedBodyOperationKind::
          WideningProductDeferredAccumulateReduceDequantizeF32) {
    config.sew = tcrv::rvv::getRVVSEW32Bits();
    config.lmul = tcrv::rvv::getRVVLMULM1();
  }
  const auto &configContract =
      tcrv::rvv::getRVVSelectedBodyConfigVLContract(config.sew, config.lmul,
                                                    config.policy);

  RVVSelectedBodyRouteAnalysis analysis;
  analysis.slice = std::move(*slice);
  analysis.description.operation = analysis.slice.arithmeticKind;
  analysis.description.memoryForm = analysis.slice.memoryForm;
  if (hasProductHead(analysis.slice))
    analysis.description.wideningProductRelation =
        productSlotRelation(analysis.slice);
  if ((analysis.slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSelect ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarDualCompareMaskAndSelect ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::F32ClampSelect ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::DequantClampF32Epilogue ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskSegment2LoadUnitStore ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd ||
      analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::
              RuntimeScalarComputedMaskIndexedGatherMAccScatter ||
      isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
          analysis.slice.arithmeticKind) ||
      isRVVSelectedBodyContractionComputedMask(
          analysis.slice.arithmeticKind) ||
      isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
          analysis.slice.arithmeticKind)) &&
      analysis.slice.compareOp)
    analysis.description.comparePredicateKind =
        analysis.slice.compareOp.getKind();
  if (analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect &&
      analysis.slice.secondaryCompareOp)
    analysis.description.secondaryComparePredicateKind =
        analysis.slice.secondaryCompareOp.getKind();
  if (analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::F32ClampSelect &&
      analysis.slice.secondaryCompareOp)
    analysis.description.secondaryComparePredicateKind =
        analysis.slice.secondaryCompareOp.getKind();
  if (analysis.slice.arithmeticKind ==
          RVVSelectedBodyOperationKind::DequantClampF32Epilogue &&
      analysis.slice.secondaryCompareOp)
    analysis.description.secondaryComparePredicateKind =
        analysis.slice.secondaryCompareOp.getKind();
  analysis.description.sew = config.sew;
  analysis.description.lmul = config.lmul;
  analysis.description.tailPolicy =
      stringifyRVVTailPolicy(config.policy.getTail());
  analysis.description.maskPolicy =
      stringifyRVVMaskPolicy(config.policy.getMask());
  analysis.description.configContractID = configContract.configContractID;
  analysis.description.runtimeVLContractID = configContract.runtimeVLContractID;
  analysis.description.runtimeAVLASource = configContract.runtimeAVLASource;
  if (analysis.slice.arithmeticKind == RVVSelectedBodyOperationKind::ReduceAdd &&
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad)
    analysis.description.runtimeControlPlanID = getRVVRuntimeAVLVLControlPlanID();
  switch (analysis.slice.memoryForm) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    analysis.description.runtimeABIOrder = configContract.runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RHSScalarBroadcast:
    break;
  case RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc:
    if (std::optional<llvm::StringRef> runtimeABIOrder =
            getExpectedRVVSelectedBodyMAccRuntimeABIOrder(
                analysis.slice.arithmeticKind))
      analysis.description.runtimeABIOrder = *runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore:
    analysis.description.runtimeABIOrder =
        kRVVRuntimeScalarSplatStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect:
    analysis.description.runtimeABIOrder =
        kRVVRuntimeScalarCompareSelectRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect:
    analysis.description.runtimeABIOrder =
        kRVVRuntimeScalarDualCompareMaskAndSelectRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect:
    analysis.description.runtimeABIOrder = kRVVF32ClampSelectRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore:
    analysis.description.runtimeABIOrder =
        kRVVRuntimeScalarComputedMaskStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore:
    analysis.description.runtimeABIOrder =
        kRVVRuntimeScalarComputedMaskLoadStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::StridedLoadStore:
    analysis.description.runtimeABIOrder = kRVVStridedRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::StridedLoadUnitStore:
    break;
  case RVVSelectedBodyMemoryForm::UnitLoadStridedStore:
    break;
  case RVVSelectedBodyMemoryForm::IndexedLoadUnitStore:
    break;
  case RVVSelectedBodyMemoryForm::UnitLoadIndexedStore:
    break;
  case RVVSelectedBodyMemoryForm::MaskedUnitLoadStore:
    break;
  case RVVSelectedBodyMemoryForm::MaskedUnitStore:
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskMemoryRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskSelectRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskStridedStoreRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore:
    analysis.description.runtimeABIOrder =
        kRVVComputedMaskStridedLoadRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore:
    analysis.description.runtimeABIOrder =
        analysis.slice.arithmeticKind ==
                RVVSelectedBodyOperationKind::
                    RuntimeScalarComputedMaskIndexedGatherLoadUnitStore
            ? llvm::StringRef(
                  kRVVRuntimeScalarComputedMaskIndexedGatherRuntimeABIOrder)
            : llvm::StringRef(kRVVComputedMaskIndexedGatherRuntimeABIOrder);
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore:
    analysis.description.runtimeABIOrder =
        analysis.slice.arithmeticKind ==
                RVVSelectedBodyOperationKind::
                    RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad
            ? llvm::StringRef(
                  kRVVRuntimeScalarComputedMaskIndexedScatterRuntimeABIOrder)
            : llvm::StringRef(kRVVComputedMaskIndexedScatterRuntimeABIOrder);
    break;
  case RVVSelectedBodyMemoryForm::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    analysis.description.runtimeABIOrder =
        kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore:
    analysis.description.runtimeABIOrder =
        analysis.slice.arithmeticKind ==
                RVVSelectedBodyOperationKind::
                    RuntimeScalarComputedMaskSegment2LoadUnitStore
            ? llvm::StringRef(
                  kRVVRuntimeScalarComputedMaskSegment2LoadRuntimeABIOrder)
            : llvm::StringRef(kRVVComputedMaskSegment2LoadRuntimeABIOrder);
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store:
    analysis.description.runtimeABIOrder =
        analysis.slice.arithmeticKind ==
                RVVSelectedBodyOperationKind::
                    RuntimeScalarComputedMaskSegment2StoreUnitLoad
            ? llvm::StringRef(
                  kRVVRuntimeScalarComputedMaskSegment2StoreRuntimeABIOrder)
            : llvm::StringRef(kRVVComputedMaskSegment2StoreRuntimeABIOrder);
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc:
    if (std::optional<llvm::StringRef> runtimeABIOrder =
            getExpectedRVVSelectedBodyMAccRuntimeABIOrder(
                analysis.slice.arithmeticKind))
      analysis.description.runtimeABIOrder = *runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc:
    if (std::optional<llvm::StringRef> runtimeABIOrder =
            getExpectedRVVSelectedBodyMAccRuntimeABIOrder(
                analysis.slice.arithmeticKind))
      analysis.description.runtimeABIOrder = *runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::
      RuntimeScalarComputedMaskUnitStrideStandaloneReduction:
    if (std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
            getRVVStandaloneReductionRouteFacts(analysis.slice.arithmeticKind))
      analysis.description.runtimeABIOrder = routeFacts->runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::Segment2LoadUnitStore:
    analysis.description.runtimeABIOrder = kRVVSegment2RuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitLoadSegment2Store:
    analysis.description.runtimeABIOrder =
        kRVVSegment2InterleaveRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction:
    if (std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
            getRVVStandaloneReductionRouteFacts(analysis.slice.arithmeticKind))
      analysis.description.runtimeABIOrder = routeFacts->runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideStandaloneReduction:
    if (std::optional<RVVStandaloneReductionRouteFacts> routeFacts =
            getRVVStandaloneReductionRouteFacts(analysis.slice.arithmeticKind))
      analysis.description.runtimeABIOrder = routeFacts->runtimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitStrideConversion:
    analysis.description.runtimeABIOrder =
        kRVVWideningConversionRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitStrideDequantization:
    analysis.description.runtimeABIOrder =
        kRVVDequantizeI32ToF32RuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue:
    analysis.description.runtimeABIOrder =
        kRVVDequantClampF32EpilogueRuntimeABIOrder;
    break;
  case RVVSelectedBodyMemoryForm::UnitStrideWideningProductReduceDequantClampF32:
    analysis.description.runtimeABIOrder =
        getRVVSelectedBodyContractionRuntimeABIOrder(
            analysis.slice.arithmeticKind);
    break;
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce:
  case RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce:
  case RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce:
    analysis.description.runtimeABIOrder = configContract.runtimeABIOrder;
    break;
  }
  if (isRVVSelectedBodyContractionRouteOperation(
          analysis.slice.arithmeticKind))
    analysis.description.runtimeABIOrder =
        getRVVSelectedBodyContractionRuntimeABIOrder(
            analysis.slice.arithmeticKind);
  if (std::optional<llvm::StringRef> runtimeABIOrder =
          getExpectedRVVSelectedBodyMAccRuntimeABIOrder(
              analysis.slice.arithmeticKind))
    analysis.description.runtimeABIOrder = *runtimeABIOrder;
  analysis.description.vlDefOpName = configContract.vlDefOpName;
  analysis.description.vlScopeOpName = configContract.vlScopeOpName;
  analysis.description.vlUses = configContract.vlUses;
  analysis.description.emitCLoopKind = configContract.emitCLoopKind;
  analysis.description.emitCLoopInductionName =
      configContract.emitCLoopInductionName;
  analysis.description.emitCFullChunkVLName =
      configContract.emitCFullChunkVLName;
  analysis.description.emitCLoopVLName =
      tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName();
  analysis.description.remainingAVLMetadata =
      configContract.remainingAVLMetadata;
  analysis.description.pointerAdvanceMetadata =
      configContract.pointerAdvanceMetadata;
  analysis.description.boundedSlice = configContract.boundedSlice;
  analysis.description.multiVL = configContract.multiVL;
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::UnitStrideConversion) {
    if (analysis.slice.arithmeticKind ==
        RVVSelectedBodyOperationKind::WidenI16ToI32) {
      analysis.description.sourceSEW = tcrv::rvv::getRVVSEW16Bits();
      analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULMF2();
      analysis.description.sourceVectorTypeName =
          getRVVSelectedBodyVectorTypeName(analysis.description.sourceSEW,
                                           analysis.description.sourceLMUL);
      analysis.description.sourceVectorCType =
          getRVVSelectedBodySignedVectorCType(analysis.description.sourceSEW,
                                              analysis.description.sourceLMUL);
      analysis.description.sourceVectorLoadIntrinsic =
          getRVVSelectedBodyVectorLoadIntrinsic(
              analysis.description.sourceSEW,
              analysis.description.sourceLMUL);
      analysis.description.conversionRelation =
          kRVVWidenI16ToI32ConversionRelation;
    } else {
      analysis.description.sourceSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
      analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULM1();
      analysis.description.sourceVectorTypeName =
          getRVVSelectedBodyVectorTypeName(analysis.description.sourceSEW,
                                           analysis.description.sourceLMUL);
      analysis.description.sourceVectorCType =
          getRVVSelectedBodySignedVectorCType(analysis.description.sourceSEW,
                                              analysis.description.sourceLMUL);
      analysis.description.sourceVectorLoadIntrinsic =
          getRVVSelectedBodyVectorLoadIntrinsic(
              analysis.description.sourceSEW,
              analysis.description.sourceLMUL);
      analysis.description.conversionRelation = kRVVWideningConversionRelation;
    }
  }
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::UnitStrideDequantization) {
    analysis.description.sourceSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
    analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULM1();
    analysis.description.sourceElementTypeName =
        getRVVSelectedBodyIntegerElementTypeName(
            analysis.description.sourceSEW);
    analysis.description.sourceVectorTypeName =
        getRVVSelectedBodyVectorTypeName(analysis.description.sourceSEW,
                                         analysis.description.sourceLMUL);
    analysis.description.sourceVectorCType =
        getRVVSelectedBodySignedVectorCType(analysis.description.sourceSEW,
                                            analysis.description.sourceLMUL);
    analysis.description.sourceVectorLoadIntrinsic =
        getRVVSelectedBodyVectorLoadIntrinsic(
            analysis.description.sourceSEW, analysis.description.sourceLMUL);
    analysis.description.resultElementTypeName =
        getRVVSelectedBodyFloatElementTypeName(
            tcrv::rvv::getRVVFirstSliceSEWBits());
    analysis.description.conversionKind = kRVVDequantizeI32ToF32Kind;
    analysis.description.dequantizationRelation =
        kRVVDequantizeI32ToF32Relation;
    analysis.description.dequantScaleRole = "dequant-scale-value";
    analysis.description.dequantScaleCType = "float";
    analysis.description.dequantScaleName = "scale";
  }
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue) {
    analysis.description.sourceSEW = tcrv::rvv::getRVVFirstSliceSEWBits();
    analysis.description.sourceLMUL = tcrv::rvv::getRVVLMULM1();
    analysis.description.sourceElementTypeName =
        getRVVSelectedBodyIntegerElementTypeName(
            analysis.description.sourceSEW);
    analysis.description.sourceVectorTypeName =
        getRVVSelectedBodyVectorTypeName(analysis.description.sourceSEW,
                                         analysis.description.sourceLMUL);
    analysis.description.sourceVectorCType =
        getRVVSelectedBodySignedVectorCType(analysis.description.sourceSEW,
                                            analysis.description.sourceLMUL);
    analysis.description.sourceVectorLoadIntrinsic =
        getRVVSelectedBodyVectorLoadIntrinsic(
            analysis.description.sourceSEW, analysis.description.sourceLMUL);
    analysis.description.resultElementTypeName =
        getRVVSelectedBodyFloatElementTypeName(
            tcrv::rvv::getRVVFirstSliceSEWBits());
    analysis.description.conversionKind = kRVVDequantizeI32ToF32Kind;
    analysis.description.dequantizationRelation =
        kRVVDequantizeI32ToF32Relation;
    analysis.description.dequantizeConvertIntrinsic =
        getRVVSelectedBodyI32ToF32DequantConvertIntrinsic();
    analysis.description.dequantizeScaleIntrinsic =
        getRVVSelectedBodyF32ScalarScaleIntrinsic();
    analysis.description.dequantScaleRole = "dequant-scale-value";
    analysis.description.dequantScaleCType = "float";
    analysis.description.dequantScaleName = "scale";
  }
  analysis.description.boundaryOpName = kRVVSelectedBodyLoweringBoundaryOpName;
  analysis.description.targetArtifactRouteID =
      getRVVSelectedBodyTargetArtifactRouteID();
  analysis.description.targetArtifactKind =
      getRVVSelectedBodyTargetArtifactKind();
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.dotLHSABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.dotRHSABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::
                 RuntimeScalarComputedMaskUnitStrideMAcc) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.dotLHSABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.dotRHSABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitStrideDequantization) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.dequantScaleABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::
                                            RuntimeScalarDualCompareMaskAndSelect) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.secondaryCompareLhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.secondaryCompareRhsScalarABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.trueValueABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.falseValueABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.sourceABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.dotRHSABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.indexABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.arithmeticKind ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedGatherLoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.sourceABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.indexABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else {
  analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::Segment2LoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field0ABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field1ABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  }
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::IndexedLoadUnitStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::UnitLoadIndexedStore)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.indexABI);
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::MaskedUnitLoadStore ||
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore)
    analysis.description.runtimeABIParameters.push_back(analysis.slice.maskABI);
  if (analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitStrideConversion &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::StridedLoadUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitLoadStridedStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::IndexedLoadUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitLoadIndexedStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::MaskedUnitLoadStore &&
      analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::MaskedUnitStore &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction &&
      analysis.slice.memoryForm !=
          RVVSelectedBodyMemoryForm::Segment2LoadUnitStore)
    analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction ||
      analysis.slice.arithmeticKind == RVVSelectedBodyOperationKind::MAccAdd)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.trueValueABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.falseValueABI);
  }
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore ||
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::
                                       ComputedMaskUnitStrideStandaloneReduction ||
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::
                                       RuntimeScalarComputedMaskUnitStrideStandaloneReduction)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.sourceABI);
  if (analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore ||
      analysis.slice.memoryForm ==
          RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore ||
      analysis.slice.arithmeticKind == RVVSelectedBodyOperationKind::
                                           RuntimeScalarComputedMaskIndexedGatherLoadUnitStore)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.indexABI);
  if (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::
                                       ComputedMaskUnitStrideStandaloneReduction)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
  if (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::
                                       RuntimeScalarComputedMaskUnitStrideStandaloneReduction)
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.accumulatorABI);
  if (analysis.slice.memoryForm ==
      RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field0ABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field1ABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field0ABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.field1ABI);
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  } else if (analysis.slice.memoryForm !=
      RVVSelectedBodyMemoryForm::Segment2LoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.runtimeElementCountABI);
  }
  }
  if (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.lhsStrideABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.rhsStrideABI);
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.outStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::StridedLoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.lhsStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadStridedStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.outStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.outStrideABI);
  } else if (analysis.slice.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore) {
    analysis.description.runtimeABIParameters.push_back(
        analysis.slice.sourceStrideABI);
  }
  llvm::Expected<RVVSelectedBodyRouteProfile> routeProfile =
      deriveRVVSelectedBodyRouteProfile(analysis.description);
  if (!routeProfile)
    return routeProfile.takeError();
  llvm::Expected<RVVSelectedBodyTypedConfigFacts> typedConfigFacts =
      deriveRVVSelectedBodyTypedConfigFacts(
          config, routeProfile->config,
          "selected RVV route analysis typed config boundary");
  if (!typedConfigFacts)
    return typedConfigFacts.takeError();
  analysis.typedConfigFacts = *typedConfigFacts;
  analysis.description.elementTypeName =
      analysis.typedConfigFacts.elementTypeName;
  llvm::Expected<RVVSelectedTargetCapabilityFacts> targetCapabilityFacts =
      collectRVVSelectedTargetCapabilityFacts(
          request.getVariant(), request.getCapabilities(),
          "selected RVV route analysis target-capability gate");
  if (!targetCapabilityFacts)
    return targetCapabilityFacts.takeError();
  if (llvm::Error error = verifyRVVSelectedTargetCapabilityForTypedConfig(
          *targetCapabilityFacts, analysis.typedConfigFacts,
          "selected RVV route analysis target-capability gate"))
    return std::move(error);
  analysis.selectedTargetCapabilityFacts = *targetCapabilityFacts;
  analysis.description.targetCapabilityProviderMirror =
      analysis.selectedTargetCapabilityFacts.providerMirror;
  analysis.description.targetCapabilityLegalityMirror =
      analysis.selectedTargetCapabilityFacts.legalityMirror;
  if (analysis.slice.arithmeticKind ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskIndexedGatherMAccScatter) {
    analysis.description.compositeGatherMAccScatterRouteFamilyPlanID =
        kRVVCompositeGatherMAccScatterRouteFamilyPlanID;
    analysis.description.compositeGatherMAccScatterTypedComputeChain =
        kRVVCompositeGatherMAccScatterTypedComputeChain;
    llvm::Expected<RVVCompositeGatherMAccScatterResourceSelection> selection =
        deriveRVVCompositeGatherMAccScatterResourceSelectionFromRealizedFacts(
            analysis.description, analysis.selectedTargetCapabilityFacts,
            analysis.slice.withVL.getOperation(),
            "selected RVV route analysis composite resource gate");
    if (!selection)
      return selection.takeError();
    analysis.description.compositeGatherMAccScatterResourceSelection =
        std::move(*selection);
  }

  llvm::Expected<RVVSelectedDispatchEnvelopeFacts> dispatchEnvelopeFacts =
      collectRVVSelectedDispatchEnvelopeFacts(
          request, "selected RVV route analysis dispatch/fallback envelope");
  if (!dispatchEnvelopeFacts)
    return dispatchEnvelopeFacts.takeError();
  analysis.description.selectedDispatchCaseMirror =
      std::move(dispatchEnvelopeFacts->selectedDispatchCaseMirror);
  analysis.description.selectedDispatchFallbackMirror =
      std::move(dispatchEnvelopeFacts->selectedDispatchFallbackMirror);
  analysis.description.lowPrecisionSelectedDispatchPolicyBoundary =
      std::move(
          dispatchEnvelopeFacts->lowPrecisionSelectedDispatchPolicyBoundary);

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> constructionRoute =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          routeProfile->operation.operationMnemonic);
  if (!constructionRoute)
    return constructionRoute.takeError();
  analysis.constructionRoute = *constructionRoute;

  // For the low-precision dequant(/clamp) routes the typed-compute-op chain is
  // derived from the ACTUAL realized structure (head op type + handoff presence),
  // so a single-scope typed body (Stage 3 flip) reports its real
  // packed_i4_nibble_unpack_product/widening_product head and omits the deleted
  // handoff -- it never asserts a phantom two-scope chain.
  const bool isDequantTypedComputeChain =
      routeProfile->operation.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      routeProfile->operation.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32 ||
      routeProfile->operation.operation ==
          RVVSelectedBodyOperationKind::
              WideningProductDeferredAccumulateReduceDequantizeF32;
  if (isDequantTypedComputeChain) {
    analysis.description.typedComputeOpName =
        getRVVSelectedBodyDequantTypedComputeOpChain(
            analysis.slice,
            routeProfile->operation.operation ==
                RVVSelectedBodyOperationKind::
                    WideningProductReduceDequantClampF32);
  } else {
    analysis.description.typedComputeOpName =
        (routeProfile->operation.operation ==
             RVVSelectedBodyOperationKind::WideningProductReduceAdd ||
         routeProfile->operation.operation ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskIndexedGatherMAccScatter)
            ? analysis.constructionRoute->typedComputeOpName
            : analysis.slice.arithmeticOp->getName().getStringRef();
  }
  analysis.description.emitCRouteID = analysis.constructionRoute->emitCRouteID;
  analysis.description.runtimeABIName =
      analysis.constructionRoute->runtimeABIName;
  analysis.description.runtimeABIContractName =
      analysis.constructionRoute->runtimeABIContractName;
  const bool isComputedMaskSegment2Load =
      routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
  const bool isRuntimeScalarComputedMaskSegment2Load =
      routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2Update =
      routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isRuntimeScalarComputedMaskSegment2Store =
      routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::
          RuntimeScalarComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2StoreLike =
      isComputedMaskSegment2Store ||
      isRuntimeScalarComputedMaskSegment2Store || isComputedMaskSegment2Update;
  const bool isComputedMaskSegment2LoadLike =
      isComputedMaskSegment2Load || isRuntimeScalarComputedMaskSegment2Load;
  const bool isWideningProductReduceDequantize =
      routeProfile->operation.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      routeProfile->operation.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool isWideningProductReduceDequantClamp =
      routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  analysis.description.vlCType = routeProfile->config.vlCType;
  analysis.description.vectorTypeName =
      isWideningProductReduceDequantize
          ? getRVVSelectedBodyFloatVectorTypeName(routeProfile->config.sew,
                                                  routeProfile->config.lmul)
          : routeProfile->config.vectorTypeName;
  analysis.description.indexVectorTypeName =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexVectorTypeName
          : "";
  analysis.description.maskTypeName =
      (routeProfile->operation.isCompareSelect ||
       routeProfile->operation.isMaskedArithmetic ||
       routeProfile->operation.isMaskedMemoryMovement ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::
               ComputedMaskStridedInputWideningDotReduceAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd ||
       isWideningProductReduceDequantClamp ||
       isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
           routeProfile->operation.operation) ||
       isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
           routeProfile->operation.operation))
          ? routeProfile->config.maskTypeName
          : "";
  analysis.description.vectorCType =
      isWideningProductReduceDequantize
          ? getRVVSelectedBodyFloatVectorCType(routeProfile->config.sew,
                                               routeProfile->config.lmul)
          : routeProfile->config.vectorCType;
  analysis.description.indexVectorCType =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexVectorCType
          : "";
  analysis.description.maskCType =
      (routeProfile->operation.isCompareSelect ||
       routeProfile->operation.isMaskedArithmetic ||
       routeProfile->operation.isMaskedMemoryMovement ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::
               ComputedMaskStridedInputWideningDotReduceAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
       routeProfile->operation.operation ==
           RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd ||
       isWideningProductReduceDequantClamp ||
       isRVVSelectedBodyComputedMaskStandaloneReductionRouteOperation(
           routeProfile->operation.operation) ||
       isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRouteOperation(
           routeProfile->operation.operation))
          ? routeProfile->config.maskCType
          : "";
  analysis.description.setVLIntrinsic =
      routeProfile->config.setVLIntrinsic;
  analysis.description.vectorLoadIntrinsic =
      routeProfile->config.vectorLoadIntrinsic;
  analysis.description.indexLoadIntrinsic =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexLoadIntrinsic
          : "";
  analysis.description.indexScaleIntrinsic =
      routeProfile->operation.isIndexedMemoryMovement
          ? routeProfile->config.indexScaleIntrinsic
          : "";
  analysis.description.indexedLoadIntrinsic =
      routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::IndexedGatherUnitStore
          ? routeProfile->config.indexedLoadIntrinsic
          : "";
  analysis.description.indexedStoreIntrinsic =
      routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::IndexedScatterUnitLoad
          ? routeProfile->config.indexedStoreIntrinsic
      : routeProfile->operation.operation ==
              RVVSelectedBodyOperationKind::
                  ComputedMaskIndexedScatterStoreUnitLoad
          ? getRVVSelectedBodyMaskedIndexedStoreIntrinsic(
                routeProfile->config.sew, routeProfile->config.lmul)
          : "";
  analysis.description.stridedLoadIntrinsic =
      (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::StridedLoadUnitStore)
          ? routeProfile->config.stridedLoadIntrinsic
          : "";
  analysis.description.maskedLoadIntrinsic =
      (analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::MaskedUnitLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore)
          ? routeProfile->targetLeaves.intrinsic
          : "";
  analysis.description.rhsBroadcastIntrinsic =
      routeProfile->targetLeaves.rhsBroadcastIntrinsic;
  analysis.description.storeIntrinsic =
      analysis.slice.memoryForm ==
              RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore
          ? routeProfile->targetLeaves.intrinsic
      : analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore
          ? getRVVSelectedBodyMaskedStoreIntrinsic(routeProfile->config.sew,
                                                   routeProfile->config.lmul)
      : isWideningProductReduceDequantize
          ? getRVVSelectedBodyFloatStoreIntrinsic(routeProfile->config.sew,
                                                  routeProfile->config.lmul)
          : routeProfile->config.storeIntrinsic;
  analysis.description.stridedStoreIntrinsic =
      (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitLoadStridedStore ||
       analysis.slice.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore)
          ? (analysis.slice.memoryForm ==
                     RVVSelectedBodyMemoryForm::
                         ComputedMaskUnitLoadStridedStore
                 ? getRVVSelectedBodyMaskedStridedStoreIntrinsic(
                       routeProfile->config.sew, routeProfile->config.lmul)
                 : routeProfile->config.stridedStoreIntrinsic)
          : "";
  analysis.description.intrinsic = routeProfile->targetLeaves.intrinsic;
  analysis.description.compareIntrinsic =
      routeProfile->targetLeaves.compareIntrinsic;
  if (routeProfile->operation.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect ||
      routeProfile->operation.operation ==
          RVVSelectedBodyOperationKind::F32ClampSelect ||
      isWideningProductReduceDequantClamp) {
    analysis.description.secondaryCompareIntrinsic =
        routeProfile->targetLeaves.compareIntrinsic;
    analysis.description.maskAndIntrinsic =
        routeProfile->operation.operation ==
                RVVSelectedBodyOperationKind::RuntimeScalarDualCompareMaskAndSelect
            ? routeProfile->targetLeaves.maskAndIntrinsic
            : llvm::StringRef();
  } else {
    analysis.description.secondaryCompareIntrinsic = "";
    analysis.description.maskAndIntrinsic = "";
  }
  analysis.description.maskedMergeIntrinsic =
      routeProfile->targetLeaves.maskedMergeIntrinsic;
  analysis.description.resultName = routeProfile->operation.resultName;
  analysis.description.maskName = routeProfile->operation.maskName;
  if (isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          routeProfile->operation.operation, analysis.slice.memoryForm)) {
    llvm::Expected<RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan>
        elementwisePlan =
            deriveRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(analysis);
    if (!elementwisePlan)
      return elementwisePlan.takeError();
    analysis.elementwiseArithmeticRouteFamilyPlan =
        std::move(*elementwisePlan);
    applyRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(
        *analysis.elementwiseArithmeticRouteFamilyPlan, analysis.description);
  }
  if (isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan>
        scalarBroadcastPlan =
            deriveRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
                analysis);
    if (!scalarBroadcastPlan)
      return scalarBroadcastPlan.takeError();
    analysis.scalarBroadcastElementwiseRouteFamilyPlan =
        std::move(*scalarBroadcastPlan);
    applyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
        *analysis.scalarBroadcastElementwiseRouteFamilyPlan,
        analysis.description);
  }
  if (isRVVSelectedBodyRuntimeScalarSplatStoreRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan>
        runtimeSplatPlan =
            deriveRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!runtimeSplatPlan)
      return runtimeSplatPlan.takeError();
    analysis.runtimeScalarSplatStoreRouteFamilyPlan =
        std::move(*runtimeSplatPlan);
    applyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan(
        *analysis.runtimeScalarSplatStoreRouteFamilyPlan,
        analysis.description);
  }
  if (isRVVSelectedBodyPlainCompareSelectRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyPlainCompareSelectRouteFamilyPlan>
        plainCompareSelectPlan =
            deriveRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!plainCompareSelectPlan)
      return plainCompareSelectPlan.takeError();
    analysis.plainCompareSelectRouteFamilyPlan =
        std::move(*plainCompareSelectPlan);
    applyRVVSelectedBodyPlainCompareSelectRouteFamilyPlan(
        *analysis.plainCompareSelectRouteFamilyPlan, analysis.description);
  }
  if (isRVVSelectedBodyWideningConversionRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyWideningConversionRouteFamilyPlan>
        wideningConversionPlan =
            deriveRVVSelectedBodyWideningConversionRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!wideningConversionPlan)
      return wideningConversionPlan.takeError();
    analysis.wideningConversionRouteFamilyPlan =
        std::move(*wideningConversionPlan);
    applyRVVSelectedBodyWideningConversionRouteFamilyPlan(
        *analysis.wideningConversionRouteFamilyPlan, analysis.description);
  }
  if (isRVVSelectedBodyDequantizationRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyDequantizationRouteFamilyPlan>
        dequantizationPlan =
            deriveRVVSelectedBodyDequantizationRouteFamilyPlan(
                analysis, routeProfile->config);
    if (!dequantizationPlan)
      return dequantizationPlan.takeError();
    analysis.dequantizationRouteFamilyPlan = std::move(*dequantizationPlan);
    applyRVVSelectedBodyDequantizationRouteFamilyPlan(
        *analysis.dequantizationRouteFamilyPlan, analysis.description);
  }
  if (isRVVSelectedBodyBaseMemoryMovementRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan>
        baseMemoryMovementPlan =
            deriveRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(analysis);
    if (!baseMemoryMovementPlan)
      return baseMemoryMovementPlan.takeError();
    analysis.baseMemoryMovementRouteFamilyPlan =
        std::move(*baseMemoryMovementPlan);
    applyRVVSelectedBodyBaseMemoryMovementRouteFamilyPlan(
        *analysis.baseMemoryMovementRouteFamilyPlan, analysis.description);
  }
  if (isRVVSelectedBodyComputedMaskSelectRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<
        RVVSelectedBodyComputedMaskSelectRouteFamilyPlan>
        computedMaskSelectPlan =
            deriveRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!computedMaskSelectPlan)
      return computedMaskSelectPlan.takeError();
    analysis.computedMaskSelectRouteFamilyPlan =
        std::move(*computedMaskSelectPlan);
    applyRVVSelectedBodyComputedMaskSelectRouteFamilyPlan(
        *analysis.computedMaskSelectRouteFamilyPlan,
        analysis.description);
  }
  if (isRVVSelectedBodyComputedMaskMemoryRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<
        RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan>
        computedMaskMemoryPlan =
            deriveRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!computedMaskMemoryPlan)
      return computedMaskMemoryPlan.takeError();
    analysis.computedMaskMemoryRouteFamilyPlan =
        std::move(*computedMaskMemoryPlan);
    applyRVVSelectedBodyComputedMaskMemoryRouteFamilyPlan(
        *analysis.computedMaskMemoryRouteFamilyPlan,
        analysis.description);
  }
  if (isRVVSelectedBodySegment2MemoryRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodySegment2MemoryRouteFamilyPlan>
        segment2MemoryPlan =
            deriveRVVSelectedBodySegment2MemoryRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!segment2MemoryPlan)
      return segment2MemoryPlan.takeError();
    analysis.segment2MemoryRouteFamilyPlan = std::move(*segment2MemoryPlan);
    applyRVVSelectedBodySegment2MemoryRouteFamilyPlan(
        *analysis.segment2MemoryRouteFamilyPlan, analysis.description);
  }
  if (isRVVSelectedBodyStandaloneReductionRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyStandaloneReductionRouteFamilyPlan>
        standaloneReductionPlan =
            deriveRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
                analysis, routeProfile->config, routeProfile->targetLeaves);
    if (!standaloneReductionPlan)
      return standaloneReductionPlan.takeError();
    analysis.standaloneReductionRouteFamilyPlan =
        std::move(*standaloneReductionPlan);
    applyRVVSelectedBodyStandaloneReductionRouteFamilyPlan(
        *analysis.standaloneReductionRouteFamilyPlan, analysis.description);
  }
  if (routeProfile->operation.isMaskedMemoryMovement) {
    const bool isComputedMask =
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedGatherLoadUnitStore ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::
                ComputedMaskIndexedScatterStoreUnitLoad ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad ||
        routeProfile->operation.operation ==
            RVVSelectedBodyOperationKind::
                RuntimeScalarComputedMaskIndexedGatherMAccScatter ||
	        routeProfile->operation.operation ==
	            RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore ||
	        routeProfile->operation.operation ==
	            RVVSelectedBodyOperationKind::
	                RuntimeScalarComputedMaskSegment2LoadUnitStore ||
	        routeProfile->operation.operation ==
	            RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
	        routeProfile->operation.operation ==
	            RVVSelectedBodyOperationKind::
	                RuntimeScalarComputedMaskSegment2StoreUnitLoad ||
	        routeProfile->operation.operation ==
	            RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
    const bool isMaskedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    const bool isRuntimeScalarComputedMaskStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
    const bool isRuntimeScalarComputedMaskLoadStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
    const bool isComputedMaskStridedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    const bool isComputedMaskIndexedGather =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGather =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedGatherLike =
        isComputedMaskIndexedGather || isRuntimeScalarComputedMaskIndexedGather ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
    const bool isRuntimeScalarComputedMaskIndexedScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
    const bool isComputedMaskIndexedScatterLike =
        isComputedMaskIndexedScatter || isRuntimeScalarComputedMaskIndexedScatter ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    analysis.description.maskRole =
        isComputedMask ? kRVVMaskedPredicateMaskRole : kRVVMaskRole;
    analysis.description.maskSource =
        isComputedMask ? kRVVMaskedCompareMaskSource : kRVVMaskSource;
    analysis.description.maskMemoryForm =
        isComputedMask ? kRVVComputedMaskMemoryMaskMemoryForm
                       : kRVVMaskMemoryForm;
    analysis.description.inactiveLaneContract =
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter
            ? kRVVMaskedIndexedStoreInactiveLaneContract
        : isComputedMaskIndexedScatterLike
            ? kRVVMaskedIndexedStoreInactiveLaneContract
        : isComputedMaskSegment2StoreLike
            ? kRVVMaskedStoreInactiveLaneContract
        : isRuntimeScalarComputedMaskStore
            ? kRVVMaskedStoreInactiveLaneContract
        : isRuntimeScalarComputedMaskLoadStore
            ? kRVVMaskedMemoryInactiveLaneContract
        : isComputedMaskStridedStore
            ? kRVVMaskedStridedStoreInactiveLaneContract
            : (isMaskedStore ? kRVVMaskedStoreInactiveLaneContract
                             : kRVVMaskedMemoryInactiveLaneContract);
    analysis.description.maskedPassthroughLayout =
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter
            ? kRVVMaskedMemoryPassthroughLayout
        : isComputedMaskIndexedScatterLike
            ? kRVVMaskedIndexedStorePassthroughLayout
        : isComputedMaskSegment2StoreLike
            ? kRVVMaskedStorePassthroughLayout
        : isRuntimeScalarComputedMaskStore
            ? kRVVMaskedStorePassthroughLayout
        : isRuntimeScalarComputedMaskLoadStore
            ? kRVVMaskedMemoryPassthroughLayout
        : isComputedMaskStridedStore
            ? kRVVMaskedStridedStorePassthroughLayout
            : (isMaskedStore ? kRVVMaskedStorePassthroughLayout
                             : kRVVMaskedMemoryPassthroughLayout);
    if (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike) {
      std::optional<RVVComputedMaskIndexedMemoryRouteFacts> routeFacts =
          getRVVComputedMaskIndexedMemoryRouteFacts(
              routeProfile->operation.operation);
      if (!routeFacts)
        return makeRVVEmitCRouteProviderError(
            "selected RVV masked indexed memory analysis requires "
            "provider-owned computed-mask indexed route facts before "
            "mask description population");
      analysis.description.maskRole = routeFacts->maskRole;
      analysis.description.maskSource = routeFacts->maskSource;
      analysis.description.maskMemoryForm = routeFacts->maskMemoryForm;
      analysis.description.inactiveLaneContract =
          routeFacts->inactiveLaneContract;
      analysis.description.maskedPassthroughLayout =
          routeFacts->maskedPassthroughLayout;
    }
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::ReduceAdd) {
    analysis.description.targetLeafProfile =
        kRVVVectorReductionTargetLeafProfile;
    analysis.description.providerSupportedMirror =
        kRVVVectorReductionProviderSupportedMirror;
    analysis.description.requiredHeaderDeclarations =
        kRVVVectorReductionRequiredHeaderDeclarations;
    analysis.description.cTypeMappingSummary =
        kRVVVectorReductionCTypeMappingSummary;
    analysis.description.reductionAccumulatorLayout =
        *analysis.slice.reduceOp.getAccumulatorLayout();
    analysis.description.reductionResultLayout =
        *analysis.slice.reduceOp.getResultLayout();
    analysis.description.reductionStoreVL = kRVVReductionStoreVL;
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::MAccAdd) {
    llvm::Expected<RVVSelectedBodyPlainMAccRouteFamilyPlan> plainMAccPlan =
        deriveRVVSelectedBodyPlainMAccRouteFamilyPlan(analysis);
    if (!plainMAccPlan)
      return plainMAccPlan.takeError();
    analysis.plainMAccRouteFamilyPlan = std::move(*plainMAccPlan);
    applyRVVSelectedBodyPlainMAccRouteFamilyPlan(
        *analysis.plainMAccRouteFamilyPlan, analysis.description);
  }
  if (routeProfile->operation.operation ==
      RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd) {
    llvm::Expected<RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan>
        scalarBroadcastMAccPlan =
            deriveRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(analysis);
    if (!scalarBroadcastMAccPlan)
      return scalarBroadcastMAccPlan.takeError();
    analysis.scalarBroadcastMAccRouteFamilyPlan =
        std::move(*scalarBroadcastMAccPlan);
    applyRVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan(
        *analysis.scalarBroadcastMAccRouteFamilyPlan, analysis.description);
  }
  if (isRVVSelectedBodyComputedMaskAccumulationRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<
        RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan>
        accumulationPlan =
            deriveRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(analysis);
    if (!accumulationPlan)
      return accumulationPlan.takeError();
    analysis.computedMaskAccumulationRouteFamilyPlan =
        std::move(*accumulationPlan);
    applyRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(
        *analysis.computedMaskAccumulationRouteFamilyPlan,
        analysis.description);
  }
  if (isRVVSelectedBodyContractionRouteOperation(
          routeProfile->operation.operation)) {
    llvm::Expected<RVVSelectedBodyContractionRouteFamilyPlan>
        contractionPlan =
            deriveRVVSelectedBodyContractionRouteFamilyPlan(analysis);
    if (!contractionPlan)
      return contractionPlan.takeError();
    analysis.contractionRouteFamilyPlan = std::move(*contractionPlan);
    applyRVVSelectedBodyContractionRouteFamilyPlan(
        *analysis.contractionRouteFamilyPlan, analysis.description);
    if (analysis.description.lowPrecisionResourceSelection.hasSelection &&
        isRVVLowPrecisionResourcePackedI4CandidateID(
            analysis.description.lowPrecisionResourceSelection
                .selectedCandidateID) &&
        analysis.description.lowPrecisionSelectedDispatchPolicyBoundary
            .hasFacts()) {
      if (llvm::Error error =
              populateRVVLowPrecisionSelectedDispatchPolicyOutput(
                  analysis.description.lowPrecisionResourceSelection,
                  analysis.description.lowPrecisionSelectedDispatchPolicyBoundary,
                  "selected RVV route analysis packed-i4 selected-dispatch "
                  "policy-output boundary"))
        return std::move(error);
    }
  }
  if (routeProfile->operation.isMemoryMovement) {
    const bool isUnitLoadStridedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::UnitLoadStridedStore;
    analysis.description.stridedMemoryLayout =
        isUnitLoadStridedStore ? kRVVUnitLoadStridedStoreMemoryLayout
                               : kRVVStridedLoadUnitStoreMemoryLayout;
    if (isUnitLoadStridedStore) {
      analysis.description.outStrideSource = kRVVDestinationByteStrideSource;
      analysis.description.sourceMemoryForm = kRVVUnitStrideSourceMemoryForm;
      analysis.description.destinationMemoryForm = "strided-store";
    } else {
      analysis.description.sourceStrideSource = kRVVSourceStrideSource;
      analysis.description.sourceMemoryForm = kRVVSourceMemoryForm;
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    }
  }
  if (routeProfile->operation.isIndexedMemoryMovement) {
    const bool isComputedMaskIndexedGather =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGather =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedGatherLike =
        isComputedMaskIndexedGather || isRuntimeScalarComputedMaskIndexedGather ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
    const bool isRuntimeScalarComputedMaskIndexedScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
    const bool isComputedMaskIndexedScatterLike =
        isComputedMaskIndexedScatter || isRuntimeScalarComputedMaskIndexedScatter ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter) {
      analysis.description.indexedMemoryLayout =
          kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterMemoryLayout;
      analysis.description.indexedWriteSideContract =
          kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterWriteSideContract;
      analysis.description.indexEEW =
          static_cast<std::int64_t>(
              analysis.slice.maskedIndexedLoadOp.getIndexEew());
      analysis.description.offsetUnit =
          analysis.slice.maskedIndexedLoadOp.getOffsetUnit();
      analysis.description.indexSource = kRVVIndexSource;
      analysis.description.indexUniqueness =
          analysis.slice.maskedIndexedStore.getIndexUniqueness();
      analysis.description.indexedDataMemoryForm =
          kRVVMaskedIndexedLoadSourceMemoryForm;
      analysis.description.indexedDestinationMemoryForm =
          kRVVMaskedIndexedStoreDestinationMemoryForm;
      analysis.description.sourceMemoryForm =
          kRVVMaskedIndexedLoadSourceMemoryForm;
      analysis.description.destinationMemoryForm =
          kRVVMaskedIndexedStoreDestinationMemoryForm;
    } else if (isComputedMaskIndexedGatherLike ||
               isComputedMaskIndexedScatterLike) {
      std::optional<RVVComputedMaskIndexedMemoryRouteFacts> routeFacts =
          getRVVComputedMaskIndexedMemoryRouteFacts(
              routeProfile->operation.operation);
      if (!routeFacts)
        return makeRVVEmitCRouteProviderError(
            "selected RVV indexed memory analysis requires provider-owned "
            "computed-mask indexed route facts before description population");
      analysis.description.indexedMemoryLayout =
          routeFacts->indexedMemoryLayout;
      analysis.description.indexedWriteSideContract =
          routeFacts->indexedWriteSideContract;
      analysis.description.indexEEW = routeFacts->indexEEW;
      analysis.description.offsetUnit = routeFacts->offsetUnit;
      analysis.description.indexSource = routeFacts->indexSource;
      analysis.description.indexUniqueness = routeFacts->indexUniqueness;
      analysis.description.indexedDataMemoryForm =
          routeFacts->indexedDataMemoryForm;
      analysis.description.indexedDestinationMemoryForm =
          routeFacts->indexedDestinationMemoryForm;
      analysis.description.sourceMemoryForm = routeFacts->sourceMemoryForm;
      analysis.description.destinationMemoryForm =
          routeFacts->destinationMemoryForm;
    } else {
      std::optional<RVVBaseMemoryMovementRouteFacts> routeFacts =
          getRVVBaseMemoryMovementRouteFacts(routeProfile->operation.operation);
      if (!routeFacts)
        return makeRVVEmitCRouteProviderError(
            "selected RVV indexed memory analysis requires provider-owned "
            "base-memory indexed route facts before description population");
      analysis.description.indexedMemoryLayout =
          routeFacts->indexedMemoryLayout;
      analysis.description.indexEEW = routeFacts->indexEEW;
      analysis.description.offsetUnit = routeFacts->offsetUnit;
      analysis.description.indexSource = routeFacts->indexSource;
      analysis.description.indexUniqueness = routeFacts->indexUniqueness;
      analysis.description.indexedDataMemoryForm =
          routeFacts->indexedDataMemoryForm;
      analysis.description.indexedDestinationMemoryForm =
          routeFacts->indexedDestinationMemoryForm;
      analysis.description.sourceMemoryForm = routeFacts->sourceMemoryForm;
      analysis.description.destinationMemoryForm =
          routeFacts->destinationMemoryForm;
    }
  }
  if (routeProfile->operation.isMaskedMemoryMovement) {
    const bool isMaskedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::MaskedUnitStore;
    const bool isRuntimeScalarComputedMaskStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore;
    const bool isRuntimeScalarComputedMaskLoadStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
    const bool isComputedMaskStridedStore =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedStore;
    const bool isComputedMaskStridedLoad =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore;
    const bool isComputedMaskIndexedGather =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGather =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherLoadUnitStore;
    const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedGatherLike =
        isComputedMaskIndexedGather || isRuntimeScalarComputedMaskIndexedGather ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    const bool isComputedMaskIndexedScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad;
    const bool isRuntimeScalarComputedMaskIndexedScatter =
        routeProfile->operation.operation ==
        RVVSelectedBodyOperationKind::
            RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad;
    const bool isComputedMaskIndexedScatterLike =
        isComputedMaskIndexedScatter || isRuntimeScalarComputedMaskIndexedScatter ||
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter;
    if (isRuntimeScalarComputedMaskStore)
      analysis.description.indexedMemoryLayout =
          kRVVRuntimeScalarComputedMaskStoreMemoryLayout;
    else if (isRuntimeScalarComputedMaskLoadStore)
      analysis.description.indexedMemoryLayout =
          kRVVRuntimeScalarComputedMaskLoadStoreMemoryLayout;
    else if (isMaskedStore)
      analysis.description.indexedMemoryLayout = kRVVMaskedStoreMemoryLayout;
    else if (isComputedMaskStridedStore)
      analysis.description.indexedMemoryLayout =
          kRVVComputedMaskStridedStoreMemoryLayout;
    else if (isComputedMaskStridedLoad)
      analysis.description.indexedMemoryLayout =
          kRVVComputedMaskStridedLoadMemoryLayout;
    else if (isComputedMaskIndexedGather)
      analysis.description.indexedMemoryLayout =
          kRVVComputedMaskIndexedGatherMemoryLayout;
    else if (isRuntimeScalarComputedMaskIndexedGather)
      analysis.description.indexedMemoryLayout =
          kRVVRuntimeScalarComputedMaskIndexedGatherMemoryLayout;
    else if (isComputedMaskIndexedScatter)
      analysis.description.indexedMemoryLayout =
          kRVVComputedMaskIndexedScatterMemoryLayout;
    else if (isRuntimeScalarComputedMaskIndexedScatter)
      analysis.description.indexedMemoryLayout =
          kRVVRuntimeScalarComputedMaskIndexedScatterMemoryLayout;
    else if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter)
      analysis.description.indexedMemoryLayout =
          kRVVRuntimeScalarComputedMaskIndexedGatherMAccScatterMemoryLayout;
    else if (isComputedMaskSegment2LoadLike)
      analysis.description.indexedMemoryLayout =
          isRuntimeScalarComputedMaskSegment2Load
              ? llvm::StringRef(
                    kRVVRuntimeScalarComputedMaskSegment2LoadMemoryLayout)
              : llvm::StringRef(kRVVComputedMaskSegment2LoadMemoryLayout);
	    else if (isComputedMaskSegment2Store)
	      analysis.description.indexedMemoryLayout =
	          kRVVComputedMaskSegment2StoreMemoryLayout;
	    else if (isRuntimeScalarComputedMaskSegment2Store)
	      analysis.description.indexedMemoryLayout =
	          kRVVRuntimeScalarComputedMaskSegment2StoreMemoryLayout;
	    else if (isComputedMaskSegment2Update)
	      analysis.description.indexedMemoryLayout =
	          kRVVComputedMaskSegment2UpdateMemoryLayout;
    else if (routeProfile->operation.operation ==
             RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore)
      analysis.description.indexedMemoryLayout = kRVVComputedMaskMemoryLayout;
    else
      analysis.description.indexedMemoryLayout = kRVVMaskedMemoryLayout;
    if (isComputedMaskStridedStore) {
      analysis.description.stridedMemoryLayout =
          kRVVComputedMaskStridedStoreMemoryLayout;
      analysis.description.outStrideSource = kRVVDestinationByteStrideSource;
    } else if (isComputedMaskStridedLoad) {
      analysis.description.stridedMemoryLayout =
          kRVVComputedMaskStridedLoadMemoryLayout;
      analysis.description.sourceStrideSource = kRVVSourceByteStrideSource;
    }
    analysis.description.sourceMemoryForm =
        isRuntimeScalarComputedMaskIndexedGatherMAccScatter
            ? kRVVMaskedIndexedLoadSourceMemoryForm
        : isRuntimeScalarComputedMaskStore ? kRVVUnitStrideSourceMemoryForm
        : isRuntimeScalarComputedMaskLoadStore ? kRVVUnitStrideSourceMemoryForm
        : isComputedMaskIndexedGatherLike
            ? kRVVMaskedIndexedLoadSourceMemoryForm
        : isComputedMaskSegment2LoadLike ? kRVVSegment2SourceMemoryForm
        : isComputedMaskSegment2StoreLike ? kRVVUnitStrideSourceMemoryForm
        : isComputedMaskIndexedScatterLike ? kRVVUnitStrideSourceMemoryForm
        : isComputedMaskStridedLoad ? kRVVMaskedStridedLoadSourceMemoryForm
                                    : kRVVUnitStrideSourceMemoryForm;
    if (isRuntimeScalarComputedMaskStore)
      analysis.description.destinationMemoryForm =
          kRVVMaskedStoreDestinationMemoryForm;
    else if (isRuntimeScalarComputedMaskLoadStore)
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    else if (isMaskedStore)
      analysis.description.destinationMemoryForm =
          kRVVMaskedStoreDestinationMemoryForm;
    else if (isComputedMaskStridedStore)
      analysis.description.destinationMemoryForm =
          kRVVMaskedStridedStoreDestinationMemoryForm;
    else if (isComputedMaskIndexedScatterLike)
      analysis.description.destinationMemoryForm =
          kRVVMaskedIndexedStoreDestinationMemoryForm;
    else if (isRuntimeScalarComputedMaskIndexedGatherMAccScatter)
      analysis.description.destinationMemoryForm =
          kRVVMaskedIndexedStoreDestinationMemoryForm;
    else if (isComputedMaskSegment2LoadLike)
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    else if (isComputedMaskSegment2StoreLike)
      analysis.description.destinationMemoryForm =
          kRVVSegment2InterleavedDestinationMemoryForm;
    else
      analysis.description.destinationMemoryForm = kRVVDestinationMemoryForm;
    if (isComputedMaskIndexedGatherLike || isComputedMaskIndexedScatterLike) {
      std::optional<RVVComputedMaskIndexedMemoryRouteFacts> routeFacts =
          getRVVComputedMaskIndexedMemoryRouteFacts(
              routeProfile->operation.operation);
      if (!routeFacts)
        return makeRVVEmitCRouteProviderError(
            "selected RVV masked indexed memory analysis requires "
            "provider-owned computed-mask indexed route facts before final "
            "description population");
      analysis.description.indexedMemoryLayout =
          routeFacts->indexedMemoryLayout;
      analysis.description.indexedWriteSideContract =
          routeFacts->indexedWriteSideContract;
      analysis.description.sourceMemoryForm = routeFacts->sourceMemoryForm;
      analysis.description.destinationMemoryForm =
          routeFacts->destinationMemoryForm;
      analysis.description.indexEEW = routeFacts->indexEEW;
      analysis.description.offsetUnit = routeFacts->offsetUnit;
      analysis.description.indexSource = routeFacts->indexSource;
      analysis.description.indexUniqueness = routeFacts->indexUniqueness;
      analysis.description.indexedDataMemoryForm =
          routeFacts->indexedDataMemoryForm;
      analysis.description.indexedDestinationMemoryForm =
          routeFacts->indexedDestinationMemoryForm;
    }
  }
  if (isRVVSelectedBodySegment2MemoryRouteOperation(
          routeProfile->operation.operation) &&
      !analysis.segment2MemoryRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        "plain segment2 memory routes require the shared route-family plan "
        "before description population");

  if (llvm::Error error = validateRVVSelectedBodyRuntimeABIParameters(
          analysis.slice, *analysis.constructionRoute,
          routeProfile->config, analysis.slice.runtimeElementCountABI,
          analysis.slice.outABI))
    return error;
  llvm::Expected<RVVRouteOperandBindingPlan> operandBindingPlan =
      deriveRVVRouteOperandBindingPlan(analysis);
  if (!operandBindingPlan)
    return operandBindingPlan.takeError();
  analysis.routeOperandBindingPlan = std::move(*operandBindingPlan);
  if (!analysis.routeOperandBindingPlan.planID.empty()) {
    analysis.description.routeOperandBindingPlanID =
        analysis.routeOperandBindingPlan.planID;
    analysis.description.routeOperandBindingSummary =
        stringifyRVVRouteOperandBindingPlan(
            analysis.routeOperandBindingPlan);
  }
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description,
          "selected RVV route analysis"))
    return std::move(error);
  llvm::Expected<std::string> execABIBindingSummary =
      summarizeRVVSelectedBodyExecABIBindings(
          request.getVariant(), analysis.description.runtimeABIParameters,
          "selected RVV route analysis exec-envelope ABI binding");
  if (!execABIBindingSummary)
    return execABIBindingSummary.takeError();
  analysis.description.execABIBindingSummary =
      std::move(*execABIBindingSummary);
  if (llvm::Error error = verifyRVVSelectedBodyConstructionRouteMapping(
          routeProfile->operation.operationMnemonic,
          analysis.description.typedComputeOpName,
          analysis.constructionRoute->emitCRouteID,
          analysis.constructionRoute->runtimeABIName))
    return std::move(error);
  if (llvm::Error error = verifySelectedRVVRoleSequence(
          analysis.slice, request, *analysis.constructionRoute))
    return std::move(error);
  if (llvm::Error error = verifyRVVSelectedBodyEmitCRouteDescription(
          analysis.description, "selected RVV EmitC route description"))
    return std::move(error);
  return analysis;
}

llvm::ArrayRef<RVVSelectedBodyOperationKind>
getRVVSelectedBodyOperationKinds() {
  return kRVVSelectedBodyOperationKinds;
}

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op) {
  if (!llvm::is_contained(kRVVSelectedBodyOperationKinds, op))
    return "<unknown-rvv-selected-body-operation>";
  return getRVVSelectedBodyOperationProfile(op).operationMnemonic;
}

llvm::StringRef
stringifyRVVSelectedBodyMemoryForm(RVVSelectedBodyMemoryForm form) {
  switch (form) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
    return "vector-rhs-load";
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    return "rhs-broadcast-load";
  case RVVSelectedBodyMemoryForm::RHSScalarBroadcast:
    return "rhs-scalar-broadcast";
  case RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc:
    return "rhs-scalar-broadcast-macc";
  case RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore:
    return "runtime-scalar-splat-store";
  case RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect:
    return "runtime-scalar-compare-select";
  case RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect:
    return "runtime-scalar-dual-cmp-mask-and-select";
  case RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect:
    return "runtime-scalar-f32-clamp-select";
  case RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore:
    return "runtime-scalar-computed-mask-store";
  case RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore:
    return "runtime-scalar-computed-mask-load-store";
  case RVVSelectedBodyMemoryForm::StridedLoadStore:
    return "strided-load-store";
  case RVVSelectedBodyMemoryForm::StridedLoadUnitStore:
    return "strided-load-unit-store";
  case RVVSelectedBodyMemoryForm::UnitLoadStridedStore:
    return "unit-load-strided-store";
  case RVVSelectedBodyMemoryForm::IndexedLoadUnitStore:
    return "indexed-load-unit-store";
  case RVVSelectedBodyMemoryForm::UnitLoadIndexedStore:
    return "unit-load-indexed-store";
  case RVVSelectedBodyMemoryForm::MaskedUnitLoadStore:
    return "masked-unit-load-store";
  case RVVSelectedBodyMemoryForm::MaskedUnitStore:
    return "masked-unit-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore:
    return "computed-mask-unit-load-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect:
    return "computed-mask-vector-select";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore:
    return "computed-mask-unit-load-strided-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc:
    return "computed-mask-unit-stride-macc";
  case RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc:
    return "runtime-scalar-computed-mask-unit-stride-macc";
  case RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore:
    return "computed-mask-strided-load-unit-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore:
    return "computed-mask-indexed-gather-load-unit-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore:
    return "computed-mask-unit-load-indexed-scatter-store";
  case RVVSelectedBodyMemoryForm::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return "runtime-scalar-computed-mask-indexed-gather-macc-scatter";
  case RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore:
    return "computed-mask-segment2-load-unit-store";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store:
    return "computed-mask-unit-load-segment2-store";
  case RVVSelectedBodyMemoryForm::Segment2LoadUnitStore:
    return "segment2-load-unit-store";
  case RVVSelectedBodyMemoryForm::UnitLoadSegment2Store:
    return "unit-load-segment2-store";
  case RVVSelectedBodyMemoryForm::UnitStrideConversion:
    return "unit-stride-conversion";
  case RVVSelectedBodyMemoryForm::UnitStrideDequantization:
    return "unit-stride-dequantization";
  case RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue:
    return "unit-stride-dequant-clamp-f32-epilogue";
  case RVVSelectedBodyMemoryForm::
      UnitStrideWideningProductReduceDequantClampF32:
    return "unit-stride-widening-product-reduce-dequant-clamp-f32";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideWideningDotReduce:
    return "computed-mask-unit-stride-widening-dot-reduce";
  case RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce:
    return "strided-input-widening-dot-reduce";
  case RVVSelectedBodyMemoryForm::ComputedMaskStridedInputWideningDotReduce:
    return "computed-mask-strided-input-widening-dot-reduce";
  case RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction:
    return "unit-stride-standalone-reduction";
  case RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideStandaloneReduction:
    return "computed-mask-unit-stride-standalone-reduction";
  case RVVSelectedBodyMemoryForm::
      RuntimeScalarComputedMaskUnitStrideStandaloneReduction:
    return "runtime-scalar-computed-mask-unit-stride-standalone-reduction";
  }
  llvm_unreachable("unknown RVV selected-body memory form");
}

llvm::StringRef
getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).emitCRouteID;
}

llvm::StringRef getRVVSelectedBodyEmissionKind() {
  return kRVVSelectedBodyEmissionKind;
}

llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName() {
  return kRVVSelectedBodyLoweringBoundaryOpName;
}

llvm::StringRef getRVVSelectedBodyRuntimeABIKind() {
  return kRVVSelectedBodyRuntimeABIKind;
}

llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).runtimeABIName;
}

llvm::StringRef getRVVSelectedBodyRuntimeGlueRole() {
  return kRVVSelectedBodyRuntimeGlueRole;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters() {
  return tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
}

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyConstructionMetadataFacts facts;
  facts.operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(description.operation);
  facts.typedComputeOpName = description.typedComputeOpName;
  facts.emitCRouteID = description.emitCRouteID;
  facts.targetArtifactRouteID = description.targetArtifactRouteID;
  facts.targetArtifactKind = description.targetArtifactKind;
  facts.runtimeABIName = description.runtimeABIName;
  facts.runtimeABIContractName = description.runtimeABIContractName;
  facts.runtimeABIParameters = description.runtimeABIParameters;
  return facts;
}

} // namespace tianchenrv::plugin::rvv
