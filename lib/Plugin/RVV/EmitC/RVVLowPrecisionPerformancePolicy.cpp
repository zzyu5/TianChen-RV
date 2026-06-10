#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/JSON.h"

#include <optional>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kPackedI4PerformancePolicyContract(
    "rvv-low-precision-packed-i4-dispatch-performance-policy.v1");
constexpr llvm::StringLiteral kPackedI4MeasurementInputContract(
    "packed-i4-same-target-performance-maturity-evidence-input.v1");
constexpr llvm::StringLiteral kPackedI4MeasurementInputAuthority(
    "measurement-evidence-input-only; provider-owned low-precision resource "
    "facts and target artifact mirrors remain the maturity contract");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementClassification(
    "regression");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementOutcomeFamily("no-win");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementBestSpeedupRange(
    "0.689815..0.705331");
constexpr llvm::StringLiteral kPackedI4Gate4TargetProfile("ssh rvv");
constexpr llvm::StringLiteral kPackedI4PerformancePreferenceDenialReason(
    "same-target-measurement-no-win-or-regression");
constexpr llvm::StringLiteral kPackedI4RouteSupportEffect(
    "preserve-executable-route-support; measurement evidence only gates "
    "performance preference and claims");
constexpr llvm::StringLiteral kPackedI4CorrectnessFallbackPolicyPath(
    "correctness-fallback");
constexpr llvm::StringLiteral kPackedI4PerformancePreferredPolicyPath(
    "performance-preferred");
constexpr llvm::StringLiteral kDispatchCaseRoleValue("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRoleValue("dispatch fallback");
constexpr llvm::StringLiteral kRVVPluginOriginValue("rvv-plugin");
constexpr llvm::StringLiteral kScalarPluginOriginValue("scalar-plugin");
constexpr llvm::StringLiteral kConservativeFallbackRoleValue("conservative");
constexpr llvm::StringLiteral kPackedI4MeasuredWinPerformanceFeedback(
    "same-target-packed-i4-measured-win.v1");
constexpr llvm::StringLiteral kPackedI4MeasuredWinPerformanceAction(
    "performance-preferred-after-same-target-win.v1");
constexpr llvm::StringLiteral kPackedI4MeasuredWinPerformanceMaturity(
    "performance-mature");
constexpr llvm::StringLiteral kPackedI4MeasuredWinPerformanceOutcome("win");
constexpr llvm::StringLiteral kPackedI4MeasuredWinRemediationDecision(
    "accepted-measured-win-performance-preferred.v1");
constexpr llvm::StringLiteral kPackedI4MeasuredWinNoBlocker("none");
constexpr std::int64_t kPackedI4Gate4MeasurementSummaryRecordCount = 12;
constexpr std::int64_t kPackedI4Gate4MeasurementRecordCount = 60;
constexpr std::int64_t kPackedI4Gate4CorrectnessRecordCount = 12;

llvm::Error makeRVVLowPrecisionPerformancePolicyError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV low-precision performance policy failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error requirePolicyString(llvm::StringRef context,
                                llvm::StringRef label,
                                llvm::StringRef actual,
                                llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires " + label + " '" + expected +
      "' but found '" + actual + "'");
}

llvm::Error requirePolicyBool(llvm::StringRef context, llvm::StringRef label,
                              bool actual, bool expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires " + label + " " +
      (expected ? "true" : "false") + " but found " +
      (actual ? "true" : "false"));
}

llvm::Error requirePolicyInt(llvm::StringRef context, llvm::StringRef label,
                             std::int64_t actual, std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires " + label + " " +
      llvm::Twine(expected) + " but found " + llvm::Twine(actual));
}

llvm::Error requirePositivePolicyInt(llvm::StringRef context,
                                     llvm::StringRef label,
                                     std::int64_t value) {
  if (value > 0)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires positive " + label + " but found " +
      llvm::Twine(value));
}

llvm::Error requireNonEmptyPolicyString(llvm::StringRef context,
                                        llvm::StringRef label,
                                        llvm::StringRef value) {
  if (!value.empty())
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires non-empty " + label);
}

llvm::Expected<std::string>
requireEvidenceInputString(const llvm::json::Object &evidenceInput,
                           llvm::StringRef context, llvm::StringRef key) {
  if (std::optional<llvm::StringRef> value = evidenceInput.getString(key))
    return value->str();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires same-target measurement record string "
                            "field '" +
      key + "'");
}

llvm::Expected<std::int64_t>
requireEvidenceInputInt(const llvm::json::Object &evidenceInput,
                        llvm::StringRef context, llvm::StringRef key) {
  if (std::optional<std::int64_t> value = evidenceInput.getInteger(key))
    return *value;
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement record integer field '" + key + "'");
}

llvm::Expected<bool>
requireEvidenceInputBool(const llvm::json::Object &evidenceInput,
                         llvm::StringRef context, llvm::StringRef key) {
  if (std::optional<bool> value = evidenceInput.getBoolean(key))
    return *value;
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement record boolean field '" + key + "'");
}

llvm::Error requireSameTargetPolicyInputTieBack(llvm::StringRef context,
                                                llvm::StringRef label,
                                                llvm::StringRef actual,
                                                llvm::StringRef expected) {
  if (expected.empty())
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) + " requires provider-owned " + label +
        " before consuming same-target measurement policy input");
  if (actual.empty())
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) + " requires same-target measurement policy input " +
        label + " tie-back '" + expected + "'");
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " rejects disconnected same-target measurement policy input " + label +
      " '" + actual + "'; provider facts require '" + expected + "'");
}

llvm::Error verifyPackedI4SelectionFacts(
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef context) {
  if (!selection.hasSelection)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires selected low-precision resource facts before dispatch/"
        "performance policy consumption");
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " accepts only provider-selected packed-i4 resource facts before "
        "applying the Gate 4 performance policy");
  if (llvm::Error error = requirePolicyString(
          context, "accepted Gate 4 packed-i4 selected candidate",
          selection.selectedCandidateID,
          kRVVLowPrecisionResourceDequantPackedI4Candidate))
    return error;
  if (!selection.isLegal ||
      selection.rejectionReason != kRVVLowPrecisionResourceNoRejectionReason)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires a legal packed-i4 resource selection with rejection reason "
        "'none' before performance policy consumption");
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 candidate set", selection.candidateSetID,
          kRVVLowPrecisionResourceCandidateSet))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 selection reason", selection.selectionReason,
          kRVVLowPrecisionResourceDequantPackedI4SelectionReason))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 operand form", selection.operandForm,
          kRVVLowPrecisionResourceOperandFormPackedI4Nibbles))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 source signedness", selection.sourceSignedness,
          kRVVLowPrecisionResourceSourceSignednessSigned))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "packed-i4 storage element width",
          selection.storageElementWidth,
          kRVVLowPrecisionResourcePackedI4StorageElementWidth))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "packed-i4 effective element width",
          selection.effectiveElementWidth,
          kRVVLowPrecisionResourcePackedI4EffectiveElementWidth))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 packing layout", selection.packingLayout,
          kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 unpack intent", selection.unpackIntent,
          kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 realization producer",
          selection.realizationProducer,
          kRVVLowPrecisionResourceRealizationProducer))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 realization decision",
          selection.realizationDecision,
          kRVVLowPrecisionResourcePackedI4RealizationDecision))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "packed-i4 realized vsetvl region count",
          selection.realizedVSetVLRegionCount,
          kRVVLowPrecisionResourcePackedI4VSetVLRegions))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "packed-i4 realized peak live vector groups",
          selection.realizedPeakLiveVectorGroups,
          kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive chain contract",
          selection.primitiveChainContractID,
          kRVVLowPrecisionResourcePrimitiveChainContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive chain kind", selection.primitiveChainKind,
          kRVVLowPrecisionResourcePrimitiveChainKind))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive widening product relation",
          selection.primitiveWideningProductRelation,
          kRVVLowPrecisionResourcePrimitiveWideningProductRelation))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive product-reduction chain relation",
          selection.primitiveProductReductionChainRelation,
          kRVVLowPrecisionResourcePrimitiveProductReductionChainRelation))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive widening product intrinsic",
          selection.primitiveWideningProductIntrinsic,
          kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsic))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive reduction intrinsic",
          selection.primitiveReductionIntrinsic,
          kRVVLowPrecisionResourcePrimitiveReductionIntrinsic))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive scalar seed splat intrinsic",
          selection.primitiveScalarSeedSplatIntrinsic,
          kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive accumulator layout",
          selection.primitiveAccumulatorLayout,
          kRVVLowPrecisionResourcePrimitiveAccumulatorLayout))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive result layout",
          selection.primitiveResultLayout,
          kRVVLowPrecisionResourcePrimitiveResultLayout))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive reduction store VL",
          selection.primitiveReductionStoreVL,
          kRVVLowPrecisionResourcePrimitiveReductionStoreVL))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 schedule decision contract",
          selection.scheduleDecisionContract,
          kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 schedule decision", selection.scheduleDecision,
          kRVVLowPrecisionResourcePackedI4ScheduleDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 schedule decision reason",
          selection.scheduleDecisionReason,
          kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "target capability provider mirror",
          selection.targetCapabilityProviderMirror))
    return error;
  return requireNonEmptyPolicyString(context, "target capability legality mirror",
                                     selection.targetCapabilityLegalityMirror);
}

llvm::Error verifyPackedI4SameTargetMeasurementPolicyInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  if (llvm::Error error = verifyPackedI4SelectionFacts(
          selection, (llvm::Twine(context) + " provider fact gate").str()))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement policy input contract", input.contract,
          kPackedI4MeasurementInputContract))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement policy input authority",
                              input.authority, kPackedI4MeasurementInputAuthority))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "measurement evidence id", input.measurementEvidenceID))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "same-target measurement evidence",
          input.sameTargetMeasurement, true))
    return error;
  if (llvm::Error error =
          requirePolicyBool(context, "ssh rvv evidence", input.sshEvidence,
                            true))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "target profile", input.targetProfile,
          kPackedI4Gate4TargetProfile))
    return error;

  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider selected candidate",
          input.providerResourceSelectedCandidate, selection.selectedCandidateID))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider route-family plan",
          input.providerResourceRouteFamilyPlan, selection.routeFamilyPlanID))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider-supported mirror", input.providerSupportedMirror,
          selection.providerSupportedMirror))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider runtime ABI order", input.providerRuntimeABIOrder,
          selection.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider schedule decision contract",
          input.providerScheduleDecisionContract,
          selection.scheduleDecisionContract))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider schedule decision",
          input.providerScheduleDecision, selection.scheduleDecision))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider schedule decision reason",
          input.providerScheduleDecisionReason,
          selection.scheduleDecisionReason))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive chain contract",
          input.providerPrimitiveChainContract,
          selection.primitiveChainContractID))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive chain kind",
          input.providerPrimitiveChainKind, selection.primitiveChainKind))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive widening-product relation",
          input.providerPrimitiveWideningProductRelation,
          selection.primitiveWideningProductRelation))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive product-reduction chain relation",
          input.providerPrimitiveProductReductionChainRelation,
          selection.primitiveProductReductionChainRelation))
    return error;

  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider remediation handoff contract",
          input.providerRemediationHandoffContract,
          selection.remediationHandoffContract))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider remediation diagnosis",
          input.providerRemediationDiagnosis, selection.remediationDiagnosis))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider remediation measurement evidence",
          input.providerRemediationMeasurementEvidence,
          selection.remediationMeasurementEvidenceID))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider remediation decision",
          input.providerRemediationDecision, selection.remediationDecision))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider remediation action", input.providerRemediationAction,
          selection.remediationAction))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider remediation dispatch preference",
          input.providerRemediationDispatchPreference,
          selection.remediationDispatchPreference))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider remediation blocker",
          input.providerRemediationBlocker, selection.remediationBlocker))
    return error;

  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "target capability provider mirror",
          input.targetCapabilityProviderMirror,
          selection.targetCapabilityProviderMirror))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "target capability legality mirror",
          input.targetCapabilityLegalityMirror,
          selection.targetCapabilityLegalityMirror))
    return error;

  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider maturity", input.providerMaturity,
          selection.performanceMaturity))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider maturity evidence", input.providerMaturityEvidence,
          selection.performanceMaturityEvidence))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider maturity outcome", input.providerMaturityOutcome,
          selection.performanceMaturityOutcome))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider performance-selection eligibility",
          input.providerPerformanceSelectionEligible,
          selection.performanceSelectionEligible))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider dispatch preference",
          input.providerDispatchPreference, selection.dispatchPreference))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider performance action",
          input.providerPerformanceAction, selection.performanceAction))
    return error;
  return requirePolicyString(context, "route support effect",
                             input.routeSupportEffect,
                             kPackedI4RouteSupportEffect);
}

RVVLowPrecisionPerformanceMeasurementOutcome
materializeRVVLowPrecisionMeasurementOutcomeFromPolicyInput(
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input) {
  RVVLowPrecisionPerformanceMeasurementOutcome outcome;
  outcome.contract = input.contract;
  outcome.measurementEvidenceID = input.measurementEvidenceID;
  outcome.measurementClassification = input.measurementClassification;
  outcome.measurementOutcomeFamily = input.measurementOutcomeFamily;
  outcome.measurementBestSpeedupRange = input.measurementBestSpeedupRange;
  outcome.measurementSummaryRecordCount = input.measurementSummaryRecordCount;
  outcome.measurementRecordCount = input.measurementRecordCount;
  outcome.correctnessRecordCount = input.correctnessRecordCount;
  outcome.sameTargetMeasurement = input.sameTargetMeasurement;
  outcome.sshEvidence = input.sshEvidence;
  outcome.targetProfile = input.targetProfile;
  outcome.providerMaturity = input.providerMaturity;
  outcome.providerMaturityEvidence = input.providerMaturityEvidence;
  outcome.providerMaturityOutcome = input.providerMaturityOutcome;
  outcome.providerPerformanceSelectionEligible =
      input.providerPerformanceSelectionEligible;
  outcome.providerDispatchPreference = input.providerDispatchPreference;
  outcome.providerPerformanceAction = input.providerPerformanceAction;
  outcome.providerScheduleDecisionContract =
      input.providerScheduleDecisionContract;
  outcome.providerScheduleDecision = input.providerScheduleDecision;
  outcome.providerScheduleDecisionReason =
      input.providerScheduleDecisionReason;
  outcome.performancePreferenceDenied = input.performancePreferenceDenied;
  outcome.performancePreferenceDenialReason =
      input.performancePreferenceDenialReason;
  outcome.performanceWinClaimAllowed = input.performanceWinClaimAllowed;
  outcome.correctnessExecutionAllowed = input.correctnessExecutionAllowed;
  outcome.providerContractUpdateRequired =
      input.providerContractUpdateRequired;
  outcome.routeSupportEffect = input.routeSupportEffect;
  return outcome;
}

RVVLowPrecisionSameTargetMeasurementPolicyInput
materializeRVVLowPrecisionPolicyInputFromMeasurementRecord(
    const RVVLowPrecisionSameTargetMeasurementRecord &record) {
  RVVLowPrecisionSameTargetMeasurementPolicyInput input;
  input.contract = record.contract;
  input.authority = record.authority;
  input.measurementEvidenceID = record.measurementEvidenceID;
  input.measurementClassification = record.measurementClassification;
  input.measurementOutcomeFamily = record.measurementOutcomeFamily;
  input.measurementBestSpeedupRange = record.measurementBestSpeedupRange;
  input.measurementSummaryRecordCount =
      record.measurementSummaryRecordCount;
  input.measurementRecordCount = record.measurementRecordCount;
  input.correctnessRecordCount = record.correctnessRecordCount;
  input.sameTargetMeasurement = record.sameTargetMeasurement;
  input.sshEvidence = record.sshEvidence;
  input.targetProfile = record.targetProfile;
  input.providerResourceSelectedCandidate =
      record.providerResourceSelectedCandidate;
  input.providerResourceRouteFamilyPlan =
      record.providerResourceRouteFamilyPlan;
  input.providerSupportedMirror = record.providerSupportedMirror;
  input.providerRuntimeABIOrder = record.providerRuntimeABIOrder;
  input.providerScheduleDecisionContract =
      record.providerScheduleDecisionContract;
  input.providerScheduleDecision = record.providerScheduleDecision;
  input.providerScheduleDecisionReason =
      record.providerScheduleDecisionReason;
  input.providerPrimitiveChainContract =
      record.providerPrimitiveChainContract;
  input.providerPrimitiveChainKind = record.providerPrimitiveChainKind;
  input.providerPrimitiveWideningProductRelation =
      record.providerPrimitiveWideningProductRelation;
  input.providerPrimitiveProductReductionChainRelation =
      record.providerPrimitiveProductReductionChainRelation;
  input.providerRemediationHandoffContract =
      record.providerRemediationHandoffContract;
  input.providerRemediationDiagnosis =
      record.providerRemediationDiagnosis;
  input.providerRemediationMeasurementEvidence =
      record.providerRemediationMeasurementEvidence;
  input.providerRemediationDecision = record.providerRemediationDecision;
  input.providerRemediationAction = record.providerRemediationAction;
  input.providerRemediationDispatchPreference =
      record.providerRemediationDispatchPreference;
  input.providerRemediationBlocker = record.providerRemediationBlocker;
  input.targetCapabilityProviderMirror =
      record.targetCapabilityProviderMirror;
  input.targetCapabilityLegalityMirror =
      record.targetCapabilityLegalityMirror;
  input.providerMaturity = record.providerMaturity;
  input.providerMaturityEvidence = record.providerMaturityEvidence;
  input.providerMaturityOutcome = record.providerMaturityOutcome;
  input.providerPerformanceSelectionEligible =
      record.providerPerformanceSelectionEligible;
  input.providerDispatchPreference = record.providerDispatchPreference;
  input.providerPerformanceAction = record.providerPerformanceAction;
  input.performancePreferenceDenied =
      record.performancePreferenceDenied;
  input.performancePreferenceDenialReason =
      record.performancePreferenceDenialReason;
  input.performanceWinClaimAllowed = record.performanceWinClaimAllowed;
  input.correctnessExecutionAllowed = record.correctnessExecutionAllowed;
  input.providerContractUpdateRequired =
      record.providerContractUpdateRequired;
  input.routeSupportEffect = record.routeSupportEffect;
  return input;
}

llvm::Error verifyPackedI4MeasurementOutcomeCommon(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  if (llvm::Error error = requirePolicyString(
          context, "measurement input contract", outcome.contract,
          kPackedI4MeasurementInputContract))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "measurement evidence id", outcome.measurementEvidenceID))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "same-target measurement evidence",
          outcome.sameTargetMeasurement, true))
    return error;
  if (llvm::Error error =
          requirePolicyBool(context, "ssh rvv evidence", outcome.sshEvidence,
                            true))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "target profile", outcome.targetProfile,
          kPackedI4Gate4TargetProfile))
    return error;
  if (outcome.measurementClassification != "win" &&
      outcome.measurementClassification != "no-win" &&
      outcome.measurementClassification != "regression")
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) + " requires measurement classification 'win', "
        "'no-win', or 'regression' but found '" +
        outcome.measurementClassification + "'");
  const llvm::StringRef expectedOutcomeFamily =
      outcome.measurementClassification == "win" ? "win" : "no-win";
  if (llvm::Error error = requirePolicyString(
          context, "measurement outcome family",
          outcome.measurementOutcomeFamily, expectedOutcomeFamily))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "measurement best-speedup range",
          outcome.measurementBestSpeedupRange))
    return error;
  if (llvm::Error error = requirePositivePolicyInt(
          context, "measurement summary record count",
          outcome.measurementSummaryRecordCount))
    return error;
  if (llvm::Error error = requirePositivePolicyInt(
          context, "measurement record count", outcome.measurementRecordCount))
    return error;
  if (llvm::Error error = requirePositivePolicyInt(
          context, "correctness record count", outcome.correctnessRecordCount))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider maturity tie-back", outcome.providerMaturity,
          selection.performanceMaturity))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider maturity evidence tie-back",
          outcome.providerMaturityEvidence,
          selection.performanceMaturityEvidence))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider maturity outcome tie-back",
          outcome.providerMaturityOutcome, selection.performanceMaturityOutcome))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider performance-selection eligibility tie-back",
          outcome.providerPerformanceSelectionEligible,
          selection.performanceSelectionEligible))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider dispatch preference tie-back",
          outcome.providerDispatchPreference, selection.dispatchPreference))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider performance action tie-back",
          outcome.providerPerformanceAction, selection.performanceAction))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider schedule decision contract tie-back",
          outcome.providerScheduleDecisionContract,
          selection.scheduleDecisionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider schedule decision tie-back",
          outcome.providerScheduleDecision, selection.scheduleDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider schedule decision reason tie-back",
          outcome.providerScheduleDecisionReason,
          selection.scheduleDecisionReason))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "correctness execution allowance",
          outcome.correctnessExecutionAllowed, true))
    return error;
  return requirePolicyString(context, "route support effect",
                             outcome.routeSupportEffect,
                             kPackedI4RouteSupportEffect);
}

llvm::Error verifyPackedI4MeasurementOutcome(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  if (llvm::Error error =
          verifyPackedI4MeasurementOutcomeCommon(selection, outcome, context))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement evidence id", outcome.measurementEvidenceID,
          kRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceID))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement classification",
          outcome.measurementClassification,
          kPackedI4Gate4MeasurementClassification))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement outcome family",
          outcome.measurementOutcomeFamily,
          kPackedI4Gate4MeasurementOutcomeFamily))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement best-speedup range",
          outcome.measurementBestSpeedupRange,
          kPackedI4Gate4MeasurementBestSpeedupRange))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "measurement summary record count",
          outcome.measurementSummaryRecordCount,
          kPackedI4Gate4MeasurementSummaryRecordCount))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "measurement record count",
          outcome.measurementRecordCount, kPackedI4Gate4MeasurementRecordCount))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "correctness record count", outcome.correctnessRecordCount,
          kPackedI4Gate4CorrectnessRecordCount))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "performance preference denial",
          outcome.performancePreferenceDenied, true))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "performance preference denial reason",
          outcome.performancePreferenceDenialReason,
          kPackedI4PerformancePreferenceDenialReason))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "performance win claim allowance",
          outcome.performanceWinClaimAllowed, false))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "provider contract update requirement",
          outcome.providerContractUpdateRequired, false))
    return error;
  return llvm::Error::success();
}

llvm::Error verifyPackedI4PerformancePreferredOutcome(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  if (llvm::Error error =
          verifyPackedI4MeasurementOutcomeCommon(selection, outcome, context))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement classification",
          outcome.measurementClassification, "win"))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement outcome family",
          outcome.measurementOutcomeFamily, "win"))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance feedback",
          selection.performanceFeedback, kPackedI4MeasuredWinPerformanceFeedback))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance baseline",
          selection.performanceBaseline,
          kRVVLowPrecisionResourcePackedI4PerformanceBaseline))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance best-speedup range",
          selection.performanceBestSpeedupRange,
          outcome.measurementBestSpeedupRange))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance action",
          selection.performanceAction, kPackedI4MeasuredWinPerformanceAction))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 remediation handoff contract",
          selection.remediationHandoffContract,
          kRVVLowPrecisionResourcePackedI4RemediationHandoffContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 remediation diagnosis",
          selection.remediationDiagnosis, "performance-preferred-measured-win"))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 remediation measurement evidence",
          selection.remediationMeasurementEvidenceID,
          outcome.measurementEvidenceID))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 remediation decision",
          selection.remediationDecision, kPackedI4MeasuredWinRemediationDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 remediation action", selection.remediationAction,
          kPackedI4MeasuredWinPerformanceAction))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 remediation dispatch preference",
          selection.remediationDispatchPreference,
          kPackedI4PerformancePreferredPolicyPath))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 remediation blocker",
          selection.remediationBlocker, kPackedI4MeasuredWinNoBlocker))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance maturity",
          selection.performanceMaturity, kPackedI4MeasuredWinPerformanceMaturity))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance maturity evidence",
          selection.performanceMaturityEvidence, outcome.measurementEvidenceID))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance maturity outcome",
          selection.performanceMaturityOutcome,
          kPackedI4MeasuredWinPerformanceOutcome))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance selection eligibility",
          selection.performanceSelectionEligible, "true"))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 dispatch preference",
          selection.dispatchPreference, kPackedI4PerformancePreferredPolicyPath))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "performance preference denial",
          outcome.performancePreferenceDenied, false))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "performance preference denial reason",
          outcome.performancePreferenceDenialReason, ""))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "performance win claim allowance",
          outcome.performanceWinClaimAllowed, true))
    return error;
  return requirePolicyBool(context, "provider contract update requirement",
                           outcome.providerContractUpdateRequired, false);
}

bool attemptsPerformancePreferredPackedI4Outcome(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome) {
  return outcome.measurementClassification == "win" ||
         outcome.measurementOutcomeFamily == "win" ||
         !outcome.performancePreferenceDenied ||
         outcome.performanceWinClaimAllowed ||
         selection.performanceSelectionEligible == "true" ||
         selection.dispatchPreference == "performance-preferred";
}

llvm::Error verifyPackedI4PolicyOutcomeConsistency(
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef context) {
  if (llvm::StringRef(selection.performanceMaturity) !=
      kRVVLowPrecisionResourcePackedI4PerformanceMaturity)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider maturity to remain executable-not-performance-"
        "mature for the accepted Gate 4 regression/no-win outcome");
  if (llvm::StringRef(selection.performanceMaturityOutcome) !=
      kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider maturity outcome 'regression' for the accepted "
        "Gate 4 no-win measurement");
  if (llvm::StringRef(selection.performanceSelectionEligible) !=
      kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider performance-selection eligibility 'false' until "
        "a newer provider contract update validates a measured win");
  if (llvm::StringRef(selection.dispatchPreference) !=
      kRVVLowPrecisionResourcePackedI4DispatchPreference)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires dispatch preference 'not-performance-preferred' for the "
        "accepted Gate 4 regression/no-win measurement");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionSelectedDispatchBoundary(
    const RVVLowPrecisionPerformancePolicyDecision &decision,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  if (!dispatchBoundary.hasSelectedDispatchCase)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires selected tcrv.exec.dispatch case facts before "
        "low-precision dispatch policy acceptance");
  if (!dispatchBoundary.hasSelectedDispatchFallback)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires selected tcrv.exec.dispatch fallback facts before "
        "low-precision dispatch policy acceptance");
  if (llvm::Error error = requirePolicyString(
          context, "selected dispatch case role",
          dispatchBoundary.selectedCaseRole, kDispatchCaseRoleValue))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "selected dispatch case origin",
          dispatchBoundary.selectedCaseOrigin, kRVVPluginOriginValue))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "selected dispatch case policy",
          dispatchBoundary.selectedCasePolicy))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "selected dispatch fallback path role",
          dispatchBoundary.fallbackPathRole, kDispatchFallbackRoleValue))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "selected dispatch fallback role",
          dispatchBoundary.fallbackRole, kConservativeFallbackRoleValue))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "selected dispatch fallback origin",
          dispatchBoundary.fallbackOrigin, kScalarPluginOriginValue))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "selected dispatch fallback policy",
          dispatchBoundary.fallbackPolicy))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "selected dispatch case mirror",
          dispatchBoundary.selectedDispatchCaseMirror))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "selected dispatch fallback mirror",
          dispatchBoundary.selectedDispatchFallbackMirror))
    return error;
  if (!decision.handoff.acceptedForDispatchPolicy)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires accepted low-precision primitive/resource/measurement "
        "handoff before selected-dispatch policy acceptance");
  if (decision.performancePreferredPathSelected) {
    if (decision.dispatchPolicyPath != kPackedI4PerformancePreferredPolicyPath)
      return makeRVVLowPrecisionPerformancePolicyError(
          llvm::Twine(context) +
          " performance-preferred selected-dispatch policy requires the "
          "performance-preferred dispatch path");
    return llvm::Error::success();
  }
  if (!decision.correctnessFallbackPathSelected ||
      decision.dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " no-win/regression selected-dispatch policy requires the "
        "correctness-fallback dispatch path");
  return llvm::Error::success();
}

} // namespace

RVVLowPrecisionSameTargetMeasurementRecord
buildRVVPackedI4Gate4SameTargetMeasurementRecord(
    const RVVLowPrecisionContractionResourceSelection &selection) {
  RVVLowPrecisionSameTargetMeasurementRecord record;
  record.contract = kPackedI4MeasurementInputContract.str();
  record.authority = kPackedI4MeasurementInputAuthority.str();
  record.measurementEvidenceID =
      kRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceID.str();
  record.measurementClassification =
      kPackedI4Gate4MeasurementClassification.str();
  record.measurementOutcomeFamily =
      kPackedI4Gate4MeasurementOutcomeFamily.str();
  record.measurementBestSpeedupRange =
      kPackedI4Gate4MeasurementBestSpeedupRange.str();
  record.measurementSummaryRecordCount =
      kPackedI4Gate4MeasurementSummaryRecordCount;
  record.measurementRecordCount = kPackedI4Gate4MeasurementRecordCount;
  record.correctnessRecordCount = kPackedI4Gate4CorrectnessRecordCount;
  record.sameTargetMeasurement = true;
  record.sshEvidence = true;
  record.targetProfile = kPackedI4Gate4TargetProfile.str();
  record.providerResourceSelectedCandidate = selection.selectedCandidateID;
  record.providerResourceRouteFamilyPlan = selection.routeFamilyPlanID;
  record.providerSupportedMirror = selection.providerSupportedMirror;
  record.providerRuntimeABIOrder = selection.runtimeABIOrder;
  record.providerScheduleDecisionContract =
      selection.scheduleDecisionContract;
  record.providerScheduleDecision = selection.scheduleDecision;
  record.providerScheduleDecisionReason =
      selection.scheduleDecisionReason;
  record.providerPrimitiveChainContract =
      selection.primitiveChainContractID;
  record.providerPrimitiveChainKind = selection.primitiveChainKind;
  record.providerPrimitiveWideningProductRelation =
      selection.primitiveWideningProductRelation;
  record.providerPrimitiveProductReductionChainRelation =
      selection.primitiveProductReductionChainRelation;
  record.providerRemediationHandoffContract =
      selection.remediationHandoffContract;
  record.providerRemediationDiagnosis = selection.remediationDiagnosis;
  record.providerRemediationMeasurementEvidence =
      selection.remediationMeasurementEvidenceID;
  record.providerRemediationDecision = selection.remediationDecision;
  record.providerRemediationAction = selection.remediationAction;
  record.providerRemediationDispatchPreference =
      selection.remediationDispatchPreference;
  record.providerRemediationBlocker = selection.remediationBlocker;
  record.targetCapabilityProviderMirror =
      selection.targetCapabilityProviderMirror;
  record.targetCapabilityLegalityMirror =
      selection.targetCapabilityLegalityMirror;
  record.providerMaturity = selection.performanceMaturity;
  record.providerMaturityEvidence = selection.performanceMaturityEvidence;
  record.providerMaturityOutcome = selection.performanceMaturityOutcome;
  record.providerPerformanceSelectionEligible =
      selection.performanceSelectionEligible;
  record.providerDispatchPreference = selection.dispatchPreference;
  record.providerPerformanceAction = selection.performanceAction;
  record.performancePreferenceDenied = true;
  record.performancePreferenceDenialReason =
      kPackedI4PerformancePreferenceDenialReason.str();
  record.performanceWinClaimAllowed = false;
  record.correctnessExecutionAllowed = true;
  record.providerContractUpdateRequired = false;
  record.routeSupportEffect = kPackedI4RouteSupportEffect.str();
  return record;
}

llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord>
buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceInput(
    const llvm::json::Object &evidenceInput, llvm::StringRef context) {
  RVVLowPrecisionSameTargetMeasurementRecord record;

  auto readString = [&](llvm::StringRef key,
                        std::string &field) -> llvm::Error {
    llvm::Expected<std::string> value =
        requireEvidenceInputString(evidenceInput, context, key);
    if (!value)
      return value.takeError();
    field = *value;
    return llvm::Error::success();
  };
  auto readInt = [&](llvm::StringRef key,
                     std::int64_t &field) -> llvm::Error {
    llvm::Expected<std::int64_t> value =
        requireEvidenceInputInt(evidenceInput, context, key);
    if (!value)
      return value.takeError();
    field = *value;
    return llvm::Error::success();
  };
  auto readBool = [&](llvm::StringRef key, bool &field) -> llvm::Error {
    llvm::Expected<bool> value =
        requireEvidenceInputBool(evidenceInput, context, key);
    if (!value)
      return value.takeError();
    field = *value;
    return llvm::Error::success();
  };

#define TCRV_READ_RECORD_STRING(Field, Key)                                    \
  if (llvm::Error error = readString(Key, record.Field))                       \
    return std::move(error)
#define TCRV_READ_RECORD_INT(Field, Key)                                       \
  if (llvm::Error error = readInt(Key, record.Field))                          \
    return std::move(error)
#define TCRV_READ_RECORD_BOOL(Field, Key)                                      \
  if (llvm::Error error = readBool(Key, record.Field))                         \
    return std::move(error)

  TCRV_READ_RECORD_STRING(contract, "contract");
  TCRV_READ_RECORD_STRING(authority, "authority");
  TCRV_READ_RECORD_STRING(measurementEvidenceID, "measurement_evidence_id");
  TCRV_READ_RECORD_STRING(measurementClassification,
                          "measurement_classification");
  TCRV_READ_RECORD_STRING(measurementOutcomeFamily,
                          "measurement_outcome_family");
  TCRV_READ_RECORD_STRING(measurementBestSpeedupRange,
                          "measurement_best_speedup_range");
  TCRV_READ_RECORD_INT(measurementSummaryRecordCount,
                       "measurement_summary_record_count");
  TCRV_READ_RECORD_INT(measurementRecordCount, "measurement_record_count");
  TCRV_READ_RECORD_INT(correctnessRecordCount, "correctness_record_count");
  TCRV_READ_RECORD_BOOL(sameTargetMeasurement, "same_target_measurement");
  TCRV_READ_RECORD_BOOL(sshEvidence, "ssh_evidence");
  TCRV_READ_RECORD_STRING(targetProfile, "target_profile");
  TCRV_READ_RECORD_STRING(providerResourceSelectedCandidate,
                          "provider_resource_selected_candidate");
  TCRV_READ_RECORD_STRING(providerResourceRouteFamilyPlan,
                          "provider_resource_route_family_plan");
  TCRV_READ_RECORD_STRING(providerSupportedMirror, "provider_supported_mirror");
  TCRV_READ_RECORD_STRING(providerRuntimeABIOrder,
                          "provider_runtime_abi_order");
  TCRV_READ_RECORD_STRING(providerScheduleDecisionContract,
                          "provider_schedule_decision_contract");
  TCRV_READ_RECORD_STRING(providerScheduleDecision,
                          "provider_schedule_decision");
  TCRV_READ_RECORD_STRING(providerScheduleDecisionReason,
                          "provider_schedule_decision_reason");
  TCRV_READ_RECORD_STRING(providerPrimitiveChainContract,
                          "provider_primitive_chain_contract");
  TCRV_READ_RECORD_STRING(providerPrimitiveChainKind,
                          "provider_primitive_chain_kind");
  TCRV_READ_RECORD_STRING(providerPrimitiveWideningProductRelation,
                          "provider_primitive_widening_product_relation");
  TCRV_READ_RECORD_STRING(
      providerPrimitiveProductReductionChainRelation,
      "provider_primitive_product_reduction_chain_relation");
  TCRV_READ_RECORD_STRING(providerRemediationHandoffContract,
                          "provider_remediation_handoff_contract");
  TCRV_READ_RECORD_STRING(providerRemediationDiagnosis,
                          "provider_remediation_diagnosis");
  TCRV_READ_RECORD_STRING(providerRemediationMeasurementEvidence,
                          "provider_remediation_measurement_evidence");
  TCRV_READ_RECORD_STRING(providerRemediationDecision,
                          "provider_remediation_decision");
  TCRV_READ_RECORD_STRING(providerRemediationAction,
                          "provider_remediation_action");
  TCRV_READ_RECORD_STRING(providerRemediationDispatchPreference,
                          "provider_remediation_dispatch_preference");
  TCRV_READ_RECORD_STRING(providerRemediationBlocker,
                          "provider_remediation_blocker");
  TCRV_READ_RECORD_STRING(targetCapabilityProviderMirror,
                          "target_capability_provider_mirror");
  TCRV_READ_RECORD_STRING(targetCapabilityLegalityMirror,
                          "target_capability_legality_mirror");
  TCRV_READ_RECORD_STRING(providerMaturity, "provider_maturity");
  TCRV_READ_RECORD_STRING(providerMaturityEvidence,
                          "provider_maturity_evidence");
  TCRV_READ_RECORD_STRING(providerMaturityOutcome, "provider_maturity_outcome");
  TCRV_READ_RECORD_STRING(providerPerformanceSelectionEligible,
                          "provider_performance_selection_eligible");
  TCRV_READ_RECORD_STRING(providerDispatchPreference,
                          "provider_dispatch_preference");
  TCRV_READ_RECORD_STRING(providerPerformanceAction,
                          "provider_performance_action");
  TCRV_READ_RECORD_BOOL(performancePreferenceDenied,
                        "performance_preference_denied");
  TCRV_READ_RECORD_STRING(performancePreferenceDenialReason,
                          "performance_preference_denial_reason");
  TCRV_READ_RECORD_BOOL(performanceWinClaimAllowed,
                        "performance_win_claim_allowed");
  TCRV_READ_RECORD_BOOL(correctnessExecutionAllowed,
                        "correctness_execution_allowed");
  TCRV_READ_RECORD_BOOL(providerContractUpdateRequired,
                        "provider_contract_update_required");
  TCRV_READ_RECORD_STRING(routeSupportEffect, "route_support_effect");

#undef TCRV_READ_RECORD_STRING
#undef TCRV_READ_RECORD_INT
#undef TCRV_READ_RECORD_BOOL

  return record;
}

RVVLowPrecisionSameTargetMeasurementPolicyInput
buildRVVLowPrecisionSameTargetMeasurementPolicyInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome) {
  RVVLowPrecisionSameTargetMeasurementPolicyInput input;
  input.contract = outcome.contract;
  input.authority = kPackedI4MeasurementInputAuthority.str();
  input.measurementEvidenceID = outcome.measurementEvidenceID;
  input.measurementClassification = outcome.measurementClassification;
  input.measurementOutcomeFamily = outcome.measurementOutcomeFamily;
  input.measurementBestSpeedupRange = outcome.measurementBestSpeedupRange;
  input.measurementSummaryRecordCount =
      outcome.measurementSummaryRecordCount;
  input.measurementRecordCount = outcome.measurementRecordCount;
  input.correctnessRecordCount = outcome.correctnessRecordCount;
  input.sameTargetMeasurement = outcome.sameTargetMeasurement;
  input.sshEvidence = outcome.sshEvidence;
  input.targetProfile = outcome.targetProfile;

  input.providerResourceSelectedCandidate = selection.selectedCandidateID;
  input.providerResourceRouteFamilyPlan = selection.routeFamilyPlanID;
  input.providerSupportedMirror = selection.providerSupportedMirror;
  input.providerRuntimeABIOrder = selection.runtimeABIOrder;
  input.providerScheduleDecisionContract =
      outcome.providerScheduleDecisionContract;
  input.providerScheduleDecision = outcome.providerScheduleDecision;
  input.providerScheduleDecisionReason =
      outcome.providerScheduleDecisionReason;
  input.providerPrimitiveChainContract = selection.primitiveChainContractID;
  input.providerPrimitiveChainKind = selection.primitiveChainKind;
  input.providerPrimitiveWideningProductRelation =
      selection.primitiveWideningProductRelation;
  input.providerPrimitiveProductReductionChainRelation =
      selection.primitiveProductReductionChainRelation;

  input.providerRemediationHandoffContract =
      selection.remediationHandoffContract;
  input.providerRemediationDiagnosis = selection.remediationDiagnosis;
  input.providerRemediationMeasurementEvidence =
      selection.remediationMeasurementEvidenceID;
  input.providerRemediationDecision = selection.remediationDecision;
  input.providerRemediationAction = selection.remediationAction;
  input.providerRemediationDispatchPreference =
      selection.remediationDispatchPreference;
  input.providerRemediationBlocker = selection.remediationBlocker;

  input.targetCapabilityProviderMirror =
      selection.targetCapabilityProviderMirror;
  input.targetCapabilityLegalityMirror =
      selection.targetCapabilityLegalityMirror;

  input.providerMaturity = outcome.providerMaturity;
  input.providerMaturityEvidence = outcome.providerMaturityEvidence;
  input.providerMaturityOutcome = outcome.providerMaturityOutcome;
  input.providerPerformanceSelectionEligible =
      outcome.providerPerformanceSelectionEligible;
  input.providerDispatchPreference = outcome.providerDispatchPreference;
  input.providerPerformanceAction = outcome.providerPerformanceAction;
  input.performancePreferenceDenied = outcome.performancePreferenceDenied;
  input.performancePreferenceDenialReason =
      outcome.performancePreferenceDenialReason;
  input.performanceWinClaimAllowed = outcome.performanceWinClaimAllowed;
  input.correctnessExecutionAllowed = outcome.correctnessExecutionAllowed;
  input.providerContractUpdateRequired =
      outcome.providerContractUpdateRequired;
  input.routeSupportEffect = outcome.routeSupportEffect;
  return input;
}

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context) {
  RVVLowPrecisionSameTargetMeasurementPolicyInput input =
      materializeRVVLowPrecisionPolicyInputFromMeasurementRecord(record);
  if (selection.hasSelection &&
      isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID) &&
      llvm::StringRef(selection.selectedCandidateID) !=
          kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return input;
  if (llvm::Error error =
          verifyPackedI4SameTargetMeasurementPolicyInput(selection, input,
                                                         context))
    return std::move(error);
  return input;
}

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceInput, llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord> record =
      buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceInput(
          evidenceInput, context);
  if (!record)
    return record.takeError();
  return buildRVVLowPrecisionSameTargetMeasurementPolicyInput(selection, *record,
                                                             context);
}

llvm::Expected<RVVLowPrecisionPerformanceMeasurementOutcome>
buildRVVLowPrecisionPerformanceMeasurementOutcomeFromSameTargetRecord(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> input =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInput(selection, record,
                                                           context);
  if (!input)
    return input.takeError();
  return materializeRVVLowPrecisionMeasurementOutcomeFromPolicyInput(*input);
}

llvm::Expected<RVVLowPrecisionPerformanceMeasurementOutcome>
consumeRVVLowPrecisionSameTargetMeasurementPolicyInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  if (llvm::Error error =
          verifyPackedI4SameTargetMeasurementPolicyInput(selection, input,
                                                         context))
    return std::move(error);
  return materializeRVVLowPrecisionMeasurementOutcomeFromPolicyInput(input);
}

llvm::StringRef stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
    RVVLowPrecisionPerformanceMeasurementDiagnosisKind kind) {
  switch (kind) {
  case RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
      CorrectnessSupportedNoWinRegression:
    return "correctness-supported-no-win-regression";
  case RVVLowPrecisionPerformanceMeasurementDiagnosisKind::StaleMeasurement:
    return "stale-measurement";
  case RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
      StaleSiblingRouteMeasurement:
    return "stale-sibling-route-measurement";
  case RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
      PerformancePreferredMeasuredWin:
    return "performance-preferred-measured-win";
  }
  return "unknown";
}

RVVLowPrecisionPerformancePolicyHandoff
diagnoseRVVLowPrecisionPerformancePolicyHandoff(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyHandoff handoff;
  handoff.handoffContract =
      kRVVLowPrecisionResourcePackedI4RemediationHandoffContract.str();
  handoff.selectedCandidateID = selection.selectedCandidateID;
  handoff.expectedSelectedCandidateID =
      kRVVLowPrecisionResourceDequantPackedI4Candidate.str();
  handoff.measurementEvidenceID = outcome.measurementEvidenceID;
  handoff.measurementClassification = outcome.measurementClassification;
  handoff.measurementOutcomeFamily = outcome.measurementOutcomeFamily;
  handoff.dispatchPreference = selection.dispatchPreference;
  handoff.performancePreferenceDenialReason =
      outcome.performancePreferenceDenialReason;

  auto classifyFailure =
      [&](RVVLowPrecisionPerformanceMeasurementDiagnosisKind kind,
          llvm::Twine reason) {
        handoff.diagnosisKind =
            stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(kind)
                .str();
        handoff.failureReason = reason.str();
        handoff.staleMeasurement =
            kind == RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                        StaleMeasurement ||
            kind == RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                        StaleSiblingRouteMeasurement;
        handoff.staleSiblingRouteMeasurement =
            kind == RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                        StaleSiblingRouteMeasurement;
        handoff.performancePreferredOutcome =
            kind == RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                        PerformancePreferredMeasuredWin;
      };

  if (selection.hasSelection &&
      isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID) &&
      llvm::StringRef(selection.selectedCandidateID) !=
          kRVVLowPrecisionResourceDequantPackedI4Candidate) {
    classifyFailure(
        RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
            StaleSiblingRouteMeasurement,
        llvm::Twine(context) +
            " diagnosed stale sibling-route measurement: accepted Gate 4 "
            "packed-i4 selected candidate '" +
            kRVVLowPrecisionResourceDequantPackedI4Candidate +
            "' cannot authorize sibling candidate '" +
            selection.selectedCandidateID + "'");
    return handoff;
  }

  llvm::Error verificationError = verifyPackedI4SelectionFacts(
      selection, (llvm::Twine(context) + " policy handoff").str());
  if (verificationError) {
    RVVLowPrecisionPerformanceMeasurementDiagnosisKind kind =
        attemptsPerformancePreferredPackedI4Outcome(selection, outcome)
            ? RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                  PerformancePreferredMeasuredWin
            : RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                  StaleMeasurement;
    classifyFailure(kind, llvm::toString(std::move(verificationError)));
    return handoff;
  }

  if (attemptsPerformancePreferredPackedI4Outcome(selection, outcome)) {
    verificationError = verifyPackedI4PerformancePreferredOutcome(
        selection, outcome, (llvm::Twine(context) + " policy handoff").str());
    if (verificationError) {
      classifyFailure(RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                          PerformancePreferredMeasuredWin,
                      llvm::toString(std::move(verificationError)));
      return handoff;
    }

    handoff.diagnosisKind =
        stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
            RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                PerformancePreferredMeasuredWin)
            .str();
    handoff.correctnessSupported = true;
    handoff.performancePreferredOutcome = true;
    handoff.acceptedForDispatchPolicy = true;
    handoff.routeSupportAllowed = selection.isLegal;
    handoff.correctnessExecutionAllowed = outcome.correctnessExecutionAllowed;
    handoff.performanceSelectionAllowed = true;
    handoff.performanceWinClaimAllowed = true;
    handoff.dispatchPreference = selection.dispatchPreference;
    handoff.performancePreferenceDenialReason =
        outcome.performancePreferenceDenialReason;
    return handoff;
  }

  verificationError = verifyPackedI4MeasurementOutcome(
      selection, outcome, (llvm::Twine(context) + " policy handoff").str());
  if (!verificationError)
    verificationError = verifyPackedI4PolicyOutcomeConsistency(
        selection, (llvm::Twine(context) + " policy handoff").str());
  if (verificationError) {
    classifyFailure(RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                        StaleMeasurement,
                    llvm::toString(std::move(verificationError)));
    return handoff;
  }

  handoff.diagnosisKind =
      stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
          RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
              CorrectnessSupportedNoWinRegression)
          .str();
  handoff.correctnessSupported = true;
  handoff.noWin = outcome.measurementOutcomeFamily == "no-win";
  handoff.regression = outcome.measurementClassification == "regression";
  handoff.acceptedForDispatchPolicy = true;
  handoff.routeSupportAllowed = selection.isLegal;
  handoff.correctnessExecutionAllowed = outcome.correctnessExecutionAllowed;
  handoff.performanceSelectionAllowed = false;
  handoff.performanceWinClaimAllowed = false;
  return handoff;
}

RVVLowPrecisionPerformancePolicyHandoff
diagnoseRVVLowPrecisionPerformancePolicyHandoff(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  if (selection.hasSelection &&
      isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID) &&
      llvm::StringRef(selection.selectedCandidateID) !=
          kRVVLowPrecisionResourceDequantPackedI4Candidate) {
    RVVLowPrecisionPerformancePolicyHandoff handoff;
    handoff.handoffContract =
        kRVVLowPrecisionResourcePackedI4RemediationHandoffContract.str();
    handoff.selectedCandidateID = selection.selectedCandidateID;
    handoff.expectedSelectedCandidateID =
        kRVVLowPrecisionResourceDequantPackedI4Candidate.str();
    handoff.measurementEvidenceID = input.measurementEvidenceID;
    handoff.measurementClassification = input.measurementClassification;
    handoff.measurementOutcomeFamily = input.measurementOutcomeFamily;
    handoff.dispatchPreference = selection.dispatchPreference;
    handoff.performancePreferenceDenialReason =
        input.performancePreferenceDenialReason;
    handoff.diagnosisKind =
        stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
            RVVLowPrecisionPerformanceMeasurementDiagnosisKind::
                StaleSiblingRouteMeasurement)
            .str();
    handoff.staleMeasurement = true;
    handoff.staleSiblingRouteMeasurement = true;
    handoff.failureReason =
        (llvm::Twine(context) +
         " diagnosed stale sibling-route measurement: accepted Gate 4 "
         "packed-i4 selected candidate '" +
         kRVVLowPrecisionResourceDequantPackedI4Candidate +
         "' cannot authorize sibling candidate '" + selection.selectedCandidateID +
         "'")
            .str();
    return handoff;
  }

  llvm::Expected<RVVLowPrecisionPerformanceMeasurementOutcome> outcome =
      consumeRVVLowPrecisionSameTargetMeasurementPolicyInput(selection, input,
                                                             context);
  if (outcome)
    return diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, *outcome,
                                                           context);

  RVVLowPrecisionPerformancePolicyHandoff handoff;
  handoff.handoffContract =
      kRVVLowPrecisionResourcePackedI4RemediationHandoffContract.str();
  handoff.selectedCandidateID = selection.selectedCandidateID;
  handoff.expectedSelectedCandidateID =
      kRVVLowPrecisionResourceDequantPackedI4Candidate.str();
  handoff.measurementEvidenceID = input.measurementEvidenceID;
  handoff.measurementClassification = input.measurementClassification;
  handoff.measurementOutcomeFamily = input.measurementOutcomeFamily;
  handoff.dispatchPreference = selection.dispatchPreference;
  handoff.performancePreferenceDenialReason =
      input.performancePreferenceDenialReason;
  handoff.diagnosisKind =
      stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
          RVVLowPrecisionPerformanceMeasurementDiagnosisKind::StaleMeasurement)
          .str();
  handoff.staleMeasurement = true;
  handoff.failureReason = llvm::toString(outcome.takeError());
  return handoff;
}

RVVLowPrecisionPerformancePolicyHandoff
diagnoseRVVLowPrecisionPerformancePolicyHandoff(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> input =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInput(selection, record,
                                                           context);
  if (input)
    return diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, *input,
                                                           context);

  RVVLowPrecisionPerformancePolicyHandoff handoff;
  handoff.handoffContract =
      kRVVLowPrecisionResourcePackedI4RemediationHandoffContract.str();
  handoff.selectedCandidateID = selection.selectedCandidateID;
  handoff.expectedSelectedCandidateID =
      kRVVLowPrecisionResourceDequantPackedI4Candidate.str();
  handoff.measurementEvidenceID = record.measurementEvidenceID;
  handoff.measurementClassification = record.measurementClassification;
  handoff.measurementOutcomeFamily = record.measurementOutcomeFamily;
  handoff.dispatchPreference = selection.dispatchPreference;
  handoff.performancePreferenceDenialReason =
      record.performancePreferenceDenialReason;
  handoff.diagnosisKind =
      stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
          RVVLowPrecisionPerformanceMeasurementDiagnosisKind::StaleMeasurement)
          .str();
  handoff.staleMeasurement = true;
  handoff.failureReason = llvm::toString(input.takeError());
  return handoff;
}

void populateRVVLowPrecisionPolicyDispatchPath(
    RVVLowPrecisionPerformancePolicyDecision &decision) {
  decision.performancePreferredPathSelected =
      decision.performanceSelectionAllowed &&
      decision.performanceWinClaimAllowed &&
      decision.dispatchPreference == kPackedI4PerformancePreferredPolicyPath;
  decision.correctnessFallbackPathSelected =
      decision.correctnessExecutionAllowed &&
      !decision.performancePreferredPathSelected;
  decision.dispatchPolicyPath =
      decision.performancePreferredPathSelected
          ? kPackedI4PerformancePreferredPolicyPath.str()
          : kPackedI4CorrectnessFallbackPolicyPath.str();
  decision.fallbackReason = decision.correctnessFallbackPathSelected
                                ? decision.performancePreferenceDenialReason
                                : "";
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyHandoff handoff =
      diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, outcome,
                                                      context);
  if (!handoff.acceptedForDispatchPolicy)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) + " policy handoff diagnosis '" +
        handoff.diagnosisKind + "' is not accepted: " +
        handoff.failureReason);

  RVVLowPrecisionPerformancePolicyDecision decision;
  decision.policyContract = kPackedI4PerformancePolicyContract.str();
  decision.handoff = std::move(handoff);
  decision.routeSupportAllowed = decision.handoff.routeSupportAllowed;
  decision.correctnessExecutionAllowed =
      decision.handoff.correctnessExecutionAllowed;
  decision.performanceSelectionAllowed =
      decision.handoff.performanceSelectionAllowed;
  decision.performanceWinClaimAllowed =
      decision.handoff.performanceWinClaimAllowed;
  decision.dispatchPreference = decision.handoff.dispatchPreference;
  decision.performancePreferenceDenialReason =
      decision.handoff.performancePreferenceDenialReason;
  populateRVVLowPrecisionPolicyDispatchPath(decision);
  return decision;
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyHandoff handoff =
      diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, input,
                                                      context);
  if (!handoff.acceptedForDispatchPolicy)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) + " policy handoff diagnosis '" +
        handoff.diagnosisKind + "' is not accepted: " +
        handoff.failureReason);

  RVVLowPrecisionPerformancePolicyDecision decision;
  decision.policyContract = kPackedI4PerformancePolicyContract.str();
  decision.handoff = std::move(handoff);
  decision.routeSupportAllowed = decision.handoff.routeSupportAllowed;
  decision.correctnessExecutionAllowed =
      decision.handoff.correctnessExecutionAllowed;
  decision.performanceSelectionAllowed =
      decision.handoff.performanceSelectionAllowed;
  decision.performanceWinClaimAllowed =
      decision.handoff.performanceWinClaimAllowed;
  decision.dispatchPreference = decision.handoff.dispatchPreference;
  decision.performancePreferenceDenialReason =
      decision.handoff.performancePreferenceDenialReason;
  populateRVVLowPrecisionPolicyDispatchPath(decision);
  return decision;
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyHandoff handoff =
      diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, record,
                                                      context);
  if (!handoff.acceptedForDispatchPolicy)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) + " policy handoff diagnosis '" +
        handoff.diagnosisKind + "' is not accepted: " +
        handoff.failureReason);

  RVVLowPrecisionPerformancePolicyDecision decision;
  decision.policyContract = kPackedI4PerformancePolicyContract.str();
  decision.handoff = std::move(handoff);
  decision.routeSupportAllowed = decision.handoff.routeSupportAllowed;
  decision.correctnessExecutionAllowed =
      decision.handoff.correctnessExecutionAllowed;
  decision.performanceSelectionAllowed =
      decision.handoff.performanceSelectionAllowed;
  decision.performanceWinClaimAllowed =
      decision.handoff.performanceWinClaimAllowed;
  decision.dispatchPreference = decision.handoff.dispatchPreference;
  decision.performancePreferenceDenialReason =
      decision.handoff.performancePreferenceDenialReason;
  populateRVVLowPrecisionPolicyDispatchPath(decision);
  return decision;
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, outcome, context);
  if (!decision)
    return decision.takeError();
  if (llvm::Error error =
          verifyRVVLowPrecisionSelectedDispatchBoundary(*decision,
                                                        dispatchBoundary,
                                                        context))
    return std::move(error);
  return decision;
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, input, context);
  if (!decision)
    return decision.takeError();
  if (llvm::Error error =
          verifyRVVLowPrecisionSelectedDispatchBoundary(*decision,
                                                        dispatchBoundary,
                                                        context))
    return std::move(error);
  return decision;
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, record, context);
  if (!decision)
    return decision.takeError();
  if (llvm::Error error =
          verifyRVVLowPrecisionSelectedDispatchBoundary(*decision,
                                                        dispatchBoundary,
                                                        context))
    return std::move(error);
  return decision;
}

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyHandoff handoff =
      diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, outcome,
                                                      context);
  RVVLowPrecisionPerformancePolicyDecision decision;
  decision.policyContract = kPackedI4PerformancePolicyContract.str();
  decision.handoff = std::move(handoff);
  if (decision.handoff.acceptedForDispatchPolicy) {
    decision.routeSupportAllowed = decision.handoff.routeSupportAllowed;
    decision.correctnessExecutionAllowed =
        decision.handoff.correctnessExecutionAllowed;
    decision.performanceSelectionAllowed =
        decision.handoff.performanceSelectionAllowed;
    decision.performanceWinClaimAllowed =
        decision.handoff.performanceWinClaimAllowed;
    decision.dispatchPreference = decision.handoff.dispatchPreference;
    decision.performancePreferenceDenialReason =
        decision.handoff.performancePreferenceDenialReason;
    populateRVVLowPrecisionPolicyDispatchPath(decision);
    return decision;
  }

  decision.routeSupportAllowed =
      selection.hasSelection && selection.isLegal &&
      selection.rejectionReason == kRVVLowPrecisionResourceNoRejectionReason;
  decision.correctnessExecutionAllowed = decision.routeSupportAllowed;
  decision.performanceSelectionAllowed = false;
  decision.performanceWinClaimAllowed = false;
  decision.dispatchPreference =
      kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
  decision.performancePreferenceDenialReason =
      decision.handoff.failureReason.empty()
          ? "missing-or-stale-low-precision-performance-evidence"
          : decision.handoff.failureReason;
  populateRVVLowPrecisionPolicyDispatchPath(decision);
  return decision;
}

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyHandoff handoff =
      diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, input,
                                                      context);
  RVVLowPrecisionPerformancePolicyDecision decision;
  decision.policyContract = kPackedI4PerformancePolicyContract.str();
  decision.handoff = std::move(handoff);
  if (decision.handoff.acceptedForDispatchPolicy) {
    decision.routeSupportAllowed = decision.handoff.routeSupportAllowed;
    decision.correctnessExecutionAllowed =
        decision.handoff.correctnessExecutionAllowed;
    decision.performanceSelectionAllowed =
        decision.handoff.performanceSelectionAllowed;
    decision.performanceWinClaimAllowed =
        decision.handoff.performanceWinClaimAllowed;
    decision.dispatchPreference = decision.handoff.dispatchPreference;
    decision.performancePreferenceDenialReason =
        decision.handoff.performancePreferenceDenialReason;
    populateRVVLowPrecisionPolicyDispatchPath(decision);
    return decision;
  }

  decision.routeSupportAllowed =
      selection.hasSelection && selection.isLegal &&
      selection.rejectionReason == kRVVLowPrecisionResourceNoRejectionReason;
  decision.correctnessExecutionAllowed = decision.routeSupportAllowed;
  decision.performanceSelectionAllowed = false;
  decision.performanceWinClaimAllowed = false;
  decision.dispatchPreference =
      kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
  decision.performancePreferenceDenialReason =
      decision.handoff.failureReason.empty()
          ? "missing-or-stale-low-precision-performance-evidence"
          : decision.handoff.failureReason;
  populateRVVLowPrecisionPolicyDispatchPath(decision);
  return decision;
}

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyHandoff handoff =
      diagnoseRVVLowPrecisionPerformancePolicyHandoff(selection, record,
                                                      context);
  RVVLowPrecisionPerformancePolicyDecision decision;
  decision.policyContract = kPackedI4PerformancePolicyContract.str();
  decision.handoff = std::move(handoff);
  if (decision.handoff.acceptedForDispatchPolicy) {
    decision.routeSupportAllowed = decision.handoff.routeSupportAllowed;
    decision.correctnessExecutionAllowed =
        decision.handoff.correctnessExecutionAllowed;
    decision.performanceSelectionAllowed =
        decision.handoff.performanceSelectionAllowed;
    decision.performanceWinClaimAllowed =
        decision.handoff.performanceWinClaimAllowed;
    decision.dispatchPreference = decision.handoff.dispatchPreference;
    decision.performancePreferenceDenialReason =
        decision.handoff.performancePreferenceDenialReason;
    populateRVVLowPrecisionPolicyDispatchPath(decision);
    return decision;
  }

  decision.routeSupportAllowed =
      selection.hasSelection && selection.isLegal &&
      selection.rejectionReason == kRVVLowPrecisionResourceNoRejectionReason;
  decision.correctnessExecutionAllowed = decision.routeSupportAllowed;
  decision.performanceSelectionAllowed = false;
  decision.performanceWinClaimAllowed = false;
  decision.dispatchPreference =
      kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
  decision.performancePreferenceDenialReason =
      decision.handoff.failureReason.empty()
          ? "missing-or-stale-low-precision-performance-evidence"
          : decision.handoff.failureReason;
  populateRVVLowPrecisionPolicyDispatchPath(decision);
  return decision;
}

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  RVVLowPrecisionPerformancePolicyDecision decision =
      resolveRVVLowPrecisionDispatchPerformancePolicy(selection, record,
                                                      context);
  if (!decision.handoff.acceptedForDispatchPolicy)
    return decision;
  if (llvm::Error error =
          verifyRVVLowPrecisionSelectedDispatchBoundary(decision,
                                                        dispatchBoundary,
                                                        context)) {
    decision.performanceSelectionAllowed = false;
    decision.performanceWinClaimAllowed = false;
    decision.performancePreferredPathSelected = false;
    decision.dispatchPreference =
        kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
    decision.performancePreferenceDenialReason =
        llvm::toString(std::move(error));
    populateRVVLowPrecisionPolicyDispatchPath(decision);
  }
  return decision;
}

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, outcome, context);
  if (!decision)
    return decision.takeError();
  if (!decision->routeSupportAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve executable route support for the accepted Gate 4 "
        "regression/no-win outcome");
  if (!decision->correctnessExecutionAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve correctness execution for the accepted Gate 4 "
        "regression/no-win outcome");
  if (decision->performancePreferredPathSelected) {
    if (decision->dispatchPolicyPath != kPackedI4PerformancePreferredPolicyPath)
      return makeRVVLowPrecisionPerformancePolicyError(
          llvm::Twine(context) +
          " accepted measured-win policy must select the "
          "performance-preferred dispatch path");
    return llvm::Error::success();
  }
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed ||
      !decision->correctnessFallbackPathSelected ||
      decision->dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must deny performance selection and win claims for the accepted "
        "Gate 4 regression/no-win outcome");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, input, context);
  if (!decision)
    return decision.takeError();
  if (!decision->routeSupportAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve executable route support for the accepted same-target "
        "measurement policy input");
  if (!decision->correctnessExecutionAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve correctness execution for the accepted same-target "
        "measurement policy input");
  if (decision->performancePreferredPathSelected) {
    if (decision->dispatchPolicyPath != kPackedI4PerformancePreferredPolicyPath)
      return makeRVVLowPrecisionPerformancePolicyError(
          llvm::Twine(context) +
          " accepted same-target measured-win input must select the "
          "performance-preferred dispatch path");
    return llvm::Error::success();
  }
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed ||
      !decision->correctnessFallbackPathSelected ||
      decision->dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " accepted same-target no-win/regression input must deny performance "
        "selection and preserve the conservative correctness fallback path");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, record, context);
  if (!decision)
    return decision.takeError();
  if (!decision->routeSupportAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve executable route support for the accepted same-target "
        "measurement record");
  if (!decision->correctnessExecutionAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve correctness execution for the accepted same-target "
        "measurement record");
  if (decision->performancePreferredPathSelected) {
    if (decision->dispatchPolicyPath != kPackedI4PerformancePreferredPolicyPath)
      return makeRVVLowPrecisionPerformancePolicyError(
          llvm::Twine(context) +
          " accepted same-target measured-win record must select the "
          "performance-preferred dispatch path");
    return llvm::Error::success();
  }
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed ||
      !decision->correctnessFallbackPathSelected ||
      decision->dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " accepted same-target no-win/regression record must deny performance "
        "selection and preserve the conservative correctness fallback path");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, outcome,
                                               dispatchBoundary, context);
  if (!decision)
    return decision.takeError();
  if (!decision->routeSupportAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve executable route support for the accepted selected-"
        "dispatch low-precision policy outcome");
  if (!decision->correctnessExecutionAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve correctness execution for the accepted selected-"
        "dispatch low-precision policy outcome");
  if (decision->performancePreferredPathSelected)
    return llvm::Error::success();
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed ||
      !decision->correctnessFallbackPathSelected ||
      decision->dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-dispatch low-precision policy must deny performance "
        "selection and preserve the conservative correctness fallback path");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, input,
                                               dispatchBoundary, context);
  if (!decision)
    return decision.takeError();
  if (!decision->routeSupportAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve executable route support for the accepted selected-"
        "dispatch same-target measurement policy input");
  if (!decision->correctnessExecutionAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve correctness execution for the accepted selected-"
        "dispatch same-target measurement policy input");
  if (decision->performancePreferredPathSelected)
    return llvm::Error::success();
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed ||
      !decision->correctnessFallbackPathSelected ||
      decision->dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-dispatch same-target measurement policy input must deny "
        "performance selection and preserve the conservative correctness "
        "fallback path");
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, record,
                                               dispatchBoundary, context);
  if (!decision)
    return decision.takeError();
  if (!decision->routeSupportAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve executable route support for the accepted selected-"
        "dispatch same-target measurement record");
  if (!decision->correctnessExecutionAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve correctness execution for the accepted selected-"
        "dispatch same-target measurement record");
  if (decision->performancePreferredPathSelected)
    return llvm::Error::success();
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed ||
      !decision->correctnessFallbackPathSelected ||
      decision->dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-dispatch same-target measurement record must deny "
        "performance selection and preserve the conservative correctness "
        "fallback path");
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
