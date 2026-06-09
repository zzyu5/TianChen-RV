#ifndef TIANCHENRV_PLUGIN_RVV_RVVLOWPRECISIONPERFORMANCEPOLICY_H
#define TIANCHENRV_PLUGIN_RVV_RVVLOWPRECISIONPERFORMANCEPOLICY_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace tianchenrv::plugin::rvv {

struct RVVLowPrecisionPerformanceMeasurementOutcome {
  std::string contract;
  std::string measurementEvidenceID;
  std::string measurementClassification;
  std::string measurementOutcomeFamily;
  std::string measurementBestSpeedupRange;
  std::int64_t measurementSummaryRecordCount = 0;
  std::int64_t measurementRecordCount = 0;
  std::int64_t correctnessRecordCount = 0;

  bool sameTargetMeasurement = false;
  bool sshEvidence = false;
  std::string targetProfile;

  std::string providerMaturity;
  std::string providerMaturityEvidence;
  std::string providerMaturityOutcome;
  std::string providerPerformanceSelectionEligible;
  std::string providerDispatchPreference;
  std::string providerPerformanceAction;

  bool performancePreferenceDenied = false;
  std::string performancePreferenceDenialReason;
  bool performanceWinClaimAllowed = false;
  bool correctnessExecutionAllowed = false;
  bool providerContractUpdateRequired = false;
  std::string routeSupportEffect;
};

enum class RVVLowPrecisionPerformanceMeasurementDiagnosisKind {
  CorrectnessSupportedNoWinRegression,
  StaleMeasurement,
  StaleSiblingRouteMeasurement,
  PerformancePreferredMeasuredWin,
};

struct RVVLowPrecisionPerformancePolicyHandoff {
  std::string handoffContract;
  std::string diagnosisKind;
  std::string selectedCandidateID;
  std::string expectedSelectedCandidateID;
  std::string measurementEvidenceID;
  std::string measurementClassification;
  std::string measurementOutcomeFamily;

  bool correctnessSupported = false;
  bool noWin = false;
  bool regression = false;
  bool staleMeasurement = false;
  bool staleSiblingRouteMeasurement = false;
  bool performancePreferredOutcome = false;
  bool acceptedForDispatchPolicy = false;

  bool routeSupportAllowed = false;
  bool correctnessExecutionAllowed = false;
  bool performanceSelectionAllowed = false;
  bool performanceWinClaimAllowed = false;
  std::string dispatchPreference;
  std::string performancePreferenceDenialReason;
  std::string failureReason;
};

struct RVVLowPrecisionPerformancePolicyDecision {
  std::string policyContract;
  RVVLowPrecisionPerformancePolicyHandoff handoff;
  bool routeSupportAllowed = false;
  bool correctnessExecutionAllowed = false;
  bool performanceSelectionAllowed = false;
  bool performanceWinClaimAllowed = false;
  bool performancePreferredPathSelected = false;
  bool correctnessFallbackPathSelected = false;
  std::string dispatchPolicyPath;
  std::string dispatchPreference;
  std::string performancePreferenceDenialReason;
  std::string fallbackReason;
};

RVVLowPrecisionPerformanceMeasurementOutcome
getAcceptedRVVPackedI4Gate4MeasurementOutcome();

llvm::StringRef stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
    RVVLowPrecisionPerformanceMeasurementDiagnosisKind kind);

RVVLowPrecisionPerformancePolicyHandoff
diagnoseRVVLowPrecisionPerformancePolicyHandoff(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVLOWPRECISIONPERFORMANCEPOLICY_H
