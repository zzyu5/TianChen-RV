#include "TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kPackedI4PerformancePolicyContract(
    "rvv-low-precision-packed-i4-dispatch-performance-policy.v1");
constexpr llvm::StringLiteral kPackedI4MeasurementInputContract(
    "packed-i4-same-target-performance-maturity-evidence-input.v1");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementEvidenceID(
    "gate4-packed-i4-real-measure-ssh/"
    "widening_product_reduce_dequantize_f32/"
    "same_target_measurement_evidence.json");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementClassification(
    "regression");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementOutcomeFamily("no-win");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementBestSpeedupRange(
    "0.688889..0.705200");
constexpr llvm::StringLiteral kPackedI4Gate4TargetProfile("ssh rvv");
constexpr llvm::StringLiteral kPackedI4PerformancePreferenceDenialReason(
    "same-target-measurement-no-win-or-regression");
constexpr llvm::StringLiteral kPackedI4RouteSupportEffect(
    "preserve-executable-route-support; measurement evidence only gates "
    "performance preference and claims");
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

llvm::Error requireNonEmptyPolicyString(llvm::StringRef context,
                                        llvm::StringRef label,
                                        llvm::StringRef value) {
  if (!value.empty())
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires non-empty " + label);
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
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "target capability provider mirror",
          selection.targetCapabilityProviderMirror))
    return error;
  return requireNonEmptyPolicyString(context, "target capability legality mirror",
                                     selection.targetCapabilityLegalityMirror);
}

llvm::Error verifyPackedI4MeasurementOutcome(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  if (llvm::Error error = requirePolicyString(
          context, "measurement input contract", outcome.contract,
          kPackedI4MeasurementInputContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "measurement evidence id", outcome.measurementEvidenceID,
          kPackedI4Gate4MeasurementEvidenceID))
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
          context, "correctness execution allowance",
          outcome.correctnessExecutionAllowed, true))
    return error;
  if (llvm::Error error = requirePolicyBool(
          context, "provider contract update requirement",
          outcome.providerContractUpdateRequired, false))
    return error;
  return requirePolicyString(context, "route support effect",
                             outcome.routeSupportEffect,
                             kPackedI4RouteSupportEffect);
}

} // namespace

RVVLowPrecisionPerformanceMeasurementOutcome
getAcceptedRVVPackedI4Gate4MeasurementOutcome() {
  RVVLowPrecisionPerformanceMeasurementOutcome outcome;
  outcome.contract = kPackedI4MeasurementInputContract.str();
  outcome.measurementEvidenceID = kPackedI4Gate4MeasurementEvidenceID.str();
  outcome.measurementClassification =
      kPackedI4Gate4MeasurementClassification.str();
  outcome.measurementOutcomeFamily =
      kPackedI4Gate4MeasurementOutcomeFamily.str();
  outcome.measurementBestSpeedupRange =
      kPackedI4Gate4MeasurementBestSpeedupRange.str();
  outcome.measurementSummaryRecordCount =
      kPackedI4Gate4MeasurementSummaryRecordCount;
  outcome.measurementRecordCount = kPackedI4Gate4MeasurementRecordCount;
  outcome.correctnessRecordCount = kPackedI4Gate4CorrectnessRecordCount;
  outcome.sameTargetMeasurement = true;
  outcome.sshEvidence = true;
  outcome.targetProfile = kPackedI4Gate4TargetProfile.str();
  outcome.providerMaturity =
      kRVVLowPrecisionResourcePackedI4PerformanceMaturity.str();
  outcome.providerMaturityEvidence =
      kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence.str();
  outcome.providerMaturityOutcome =
      kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome.str();
  outcome.providerPerformanceSelectionEligible =
      kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible.str();
  outcome.providerDispatchPreference =
      kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
  outcome.providerPerformanceAction =
      kRVVLowPrecisionResourcePackedI4PerformanceAction.str();
  outcome.performancePreferenceDenied = true;
  outcome.performancePreferenceDenialReason =
      kPackedI4PerformancePreferenceDenialReason.str();
  outcome.performanceWinClaimAllowed = false;
  outcome.correctnessExecutionAllowed = true;
  outcome.providerContractUpdateRequired = false;
  outcome.routeSupportEffect = kPackedI4RouteSupportEffect.str();
  return outcome;
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context) {
  if (llvm::Error error = verifyPackedI4SelectionFacts(selection, context))
    return std::move(error);
  if (llvm::Error error =
          verifyPackedI4MeasurementOutcome(selection, outcome, context))
    return std::move(error);

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

  RVVLowPrecisionPerformancePolicyDecision decision;
  decision.policyContract = kPackedI4PerformancePolicyContract.str();
  decision.routeSupportAllowed = selection.isLegal;
  decision.correctnessExecutionAllowed = outcome.correctnessExecutionAllowed;
  decision.performanceSelectionAllowed = false;
  decision.performanceWinClaimAllowed = false;
  decision.dispatchPreference = selection.dispatchPreference;
  decision.performancePreferenceDenialReason =
      outcome.performancePreferenceDenialReason;
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
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must deny performance selection and win claims for the accepted "
        "Gate 4 regression/no-win outcome");
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
