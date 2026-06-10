#ifndef TIANCHENRV_PLUGIN_RVV_RVVLOWPRECISIONPERFORMANCEPOLICY_H
#define TIANCHENRV_PLUGIN_RVV_RVVLOWPRECISIONPERFORMANCEPOLICY_H

#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace llvm::json {
class Object;
} // namespace llvm::json

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
  std::string providerScheduleDecisionContract;
  std::string providerScheduleDecision;
  std::string providerScheduleDecisionReason;
  std::string providerResourceCostContract;
  std::string providerResourceCostModel;
  std::int64_t providerResourceCostLoopBodySteps = 0;
  std::string providerResourceCostBlocker;
  std::string providerPerformanceAdmissionDecision;
  std::string providerPerformanceAdmissionClosure;
  std::string providerPerformanceAdmissionReopenRequirement;
  std::string providerBeyondLocalRepairAdmissionContract;
  std::string providerBeyondLocalRepairAdmissionDecision;
  std::string providerBeyondLocalRepairAdmissionBlocker;
  std::string providerBeyondLocalRepairAdmissionReopenRequirement;
  std::string providerRealizationAdmissionContract;
  std::string providerRealizationAdmissionDecision;
  std::string providerRealizationAdmissionEvidence;
  std::string providerRealizationAdmissionDispatchPolicy;
  std::string providerRealizationAdmissionScheduleDecisionContract;
  std::string providerRealizationAdmissionScheduleDecision;
  std::string providerRealizationAdmissionScheduleDecisionReason;

  bool performancePreferenceDenied = false;
  std::string performancePreferenceDenialReason;
  bool performanceWinClaimAllowed = false;
  bool correctnessExecutionAllowed = false;
  bool providerContractUpdateRequired = false;
  std::string routeSupportEffect;
};

struct RVVLowPrecisionSameTargetMeasurementRecord {
  std::string contract;
  std::string authority;
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
  std::string sourceRecordContract;
  std::string sourceSelectedVariant;
  std::string sourceSelectedInput;
  std::string sourceGeneratedFunction;
  std::string generatedArtifactIdentityContract;
  std::string generatedArtifactObjectPath;
  std::string generatedArtifactObjectSHA256;
  std::string generatedArtifactHeaderPath;
  std::string generatedArtifactHeaderSHA256;
  std::string measurementTarget;
  std::string measurementTargetProvenance;
  std::string measurementRuntimeCountSet;
  std::string measurementRuntimeCountProvenance;
  std::string pressureProfileLabel;
  std::string pressureProfileLabelProvenance;

  std::string providerResourceSelectedCandidate;
  std::string providerResourcePlanningContract;
  std::string providerResourceOperandForm;
  std::string providerResourceSourceSignedness;
  std::int64_t providerResourceStorageElementWidth = 0;
  std::int64_t providerResourceEffectiveElementWidth = 0;
  std::string providerResourcePackingLayout;
  std::string providerResourceUnpackIntent;
  std::int64_t providerResourceVSetVLRegionCount = 0;
  std::string providerRuntimeAVLSource;
  std::string providerResourceRouteFamilyPlan;
  std::string providerSupportedMirror;
  std::string providerRuntimeABIOrder;
  std::string providerScheduleDecisionContract;
  std::string providerScheduleDecision;
  std::string providerScheduleDecisionReason;
  std::string providerResourceCostContract;
  std::string providerResourceCostModel;
  std::int64_t providerResourceCostLoopBodySteps = 0;
  std::string providerResourceCostBlocker;
  std::string providerPerformanceAdmissionDecision;
  std::string providerPerformanceAdmissionClosure;
  std::string providerPerformanceAdmissionReopenRequirement;
  std::string providerBeyondLocalRepairAdmissionContract;
  std::string providerBeyondLocalRepairAdmissionDecision;
  std::string providerBeyondLocalRepairAdmissionBlocker;
  std::string providerBeyondLocalRepairAdmissionReopenRequirement;
  std::string providerRealizationAdmissionContract;
  std::string providerRealizationAdmissionDecision;
  std::string providerRealizationAdmissionEvidence;
  std::string providerRealizationAdmissionDispatchPolicy;
  std::string providerRealizationAdmissionScheduleDecisionContract;
  std::string providerRealizationAdmissionScheduleDecision;
  std::string providerRealizationAdmissionScheduleDecisionReason;
  std::string providerPrimitiveChainContract;
  std::string providerPrimitiveChainKind;
  std::string providerPrimitiveContract;
  std::string providerPrimitiveKind;
  std::string providerWideningProductMultiplicandRoles;
  std::string providerWideningProductExtensionPolicy;
  std::string providerPrimitiveSourceLoad;
  std::string providerPrimitiveSourceExtension;
  std::string providerPrimitiveSourceDType;
  std::string providerPrimitiveSourceSignedness;
  std::int64_t providerPrimitiveSourceSEW = 0;
  std::string providerPrimitiveSourceLMUL;
  std::string providerPrimitiveProductDType;
  std::int64_t providerPrimitiveProductSEW = 0;
  std::string providerPrimitiveProductLMUL;
  std::string providerPrimitiveAccumulatorDType;
  std::int64_t providerPrimitiveAccumulatorSEW = 0;
  std::string providerPrimitiveAccumulatorLMUL;
  std::string providerPrimitiveResultDType;
  std::int64_t providerPrimitiveResultSEW = 0;
  std::string providerPrimitiveResultLMUL;
  std::string providerPrimitiveWideningProductRelation;
  std::string providerPrimitiveProductReductionChainRelation;
  std::string providerPrimitiveWideningProductIntrinsic;
  std::string providerPrimitiveReductionIntrinsic;
  std::string providerPrimitiveScalarSeedSplatIntrinsic;
  std::string providerPrimitiveAccumulatorLayout;
  std::string providerPrimitiveResultLayout;
  std::string providerPrimitiveReductionStoreVL;

  std::string providerRemediationHandoffContract;
  std::string providerRemediationDiagnosis;
  std::string providerRemediationMeasurementEvidence;
  std::string providerRemediationDecision;
  std::string providerRemediationAction;
  std::string providerRemediationDispatchPreference;
  std::string providerRemediationBlocker;

  std::string targetCapabilityProviderMirror;
  std::string targetCapabilityLegalityMirror;

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

struct RVVLowPrecisionSameTargetMeasurementPolicyInput {
  std::string contract;
  std::string authority;
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
  std::string sourceRecordContract;
  std::string sourceSelectedVariant;
  std::string sourceSelectedInput;
  std::string sourceGeneratedFunction;
  std::string generatedArtifactIdentityContract;
  std::string generatedArtifactObjectPath;
  std::string generatedArtifactObjectSHA256;
  std::string generatedArtifactHeaderPath;
  std::string generatedArtifactHeaderSHA256;
  std::string measurementTarget;
  std::string measurementTargetProvenance;
  std::string measurementRuntimeCountSet;
  std::string measurementRuntimeCountProvenance;
  std::string pressureProfileLabel;
  std::string pressureProfileLabelProvenance;

  std::string providerResourceSelectedCandidate;
  std::string providerResourcePlanningContract;
  std::string providerResourceOperandForm;
  std::string providerResourceSourceSignedness;
  std::int64_t providerResourceStorageElementWidth = 0;
  std::int64_t providerResourceEffectiveElementWidth = 0;
  std::string providerResourcePackingLayout;
  std::string providerResourceUnpackIntent;
  std::int64_t providerResourceVSetVLRegionCount = 0;
  std::string providerRuntimeAVLSource;
  std::string providerResourceRouteFamilyPlan;
  std::string providerSupportedMirror;
  std::string providerRuntimeABIOrder;
  std::string providerScheduleDecisionContract;
  std::string providerScheduleDecision;
  std::string providerScheduleDecisionReason;
  std::string providerResourceCostContract;
  std::string providerResourceCostModel;
  std::int64_t providerResourceCostLoopBodySteps = 0;
  std::string providerResourceCostBlocker;
  std::string providerPerformanceAdmissionDecision;
  std::string providerPerformanceAdmissionClosure;
  std::string providerPerformanceAdmissionReopenRequirement;
  std::string providerBeyondLocalRepairAdmissionContract;
  std::string providerBeyondLocalRepairAdmissionDecision;
  std::string providerBeyondLocalRepairAdmissionBlocker;
  std::string providerBeyondLocalRepairAdmissionReopenRequirement;
  std::string providerRealizationAdmissionContract;
  std::string providerRealizationAdmissionDecision;
  std::string providerRealizationAdmissionEvidence;
  std::string providerRealizationAdmissionDispatchPolicy;
  std::string providerRealizationAdmissionScheduleDecisionContract;
  std::string providerRealizationAdmissionScheduleDecision;
  std::string providerRealizationAdmissionScheduleDecisionReason;
  std::string providerPrimitiveChainContract;
  std::string providerPrimitiveChainKind;
  std::string providerPrimitiveContract;
  std::string providerPrimitiveKind;
  std::string providerWideningProductMultiplicandRoles;
  std::string providerWideningProductExtensionPolicy;
  std::string providerPrimitiveSourceLoad;
  std::string providerPrimitiveSourceExtension;
  std::string providerPrimitiveSourceDType;
  std::string providerPrimitiveSourceSignedness;
  std::int64_t providerPrimitiveSourceSEW = 0;
  std::string providerPrimitiveSourceLMUL;
  std::string providerPrimitiveProductDType;
  std::int64_t providerPrimitiveProductSEW = 0;
  std::string providerPrimitiveProductLMUL;
  std::string providerPrimitiveAccumulatorDType;
  std::int64_t providerPrimitiveAccumulatorSEW = 0;
  std::string providerPrimitiveAccumulatorLMUL;
  std::string providerPrimitiveResultDType;
  std::int64_t providerPrimitiveResultSEW = 0;
  std::string providerPrimitiveResultLMUL;
  std::string providerPrimitiveWideningProductRelation;
  std::string providerPrimitiveProductReductionChainRelation;
  std::string providerPrimitiveWideningProductIntrinsic;
  std::string providerPrimitiveReductionIntrinsic;
  std::string providerPrimitiveScalarSeedSplatIntrinsic;
  std::string providerPrimitiveAccumulatorLayout;
  std::string providerPrimitiveResultLayout;
  std::string providerPrimitiveReductionStoreVL;

  std::string providerRemediationHandoffContract;
  std::string providerRemediationDiagnosis;
  std::string providerRemediationMeasurementEvidence;
  std::string providerRemediationDecision;
  std::string providerRemediationAction;
  std::string providerRemediationDispatchPreference;
  std::string providerRemediationBlocker;

  std::string targetCapabilityProviderMirror;
  std::string targetCapabilityLegalityMirror;

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

struct RVVLowPrecisionProductionPressureProfile {
  std::string contract;
  std::string authority;
  std::string pressurePath;

  std::string selectedCandidateID;
  std::string measurementProviderCandidateID;
  std::string resourcePlanningContract;
  std::string resourceOperandForm;
  std::string resourceSourceSignedness;
  std::int64_t resourceStorageElementWidth = 0;
  std::int64_t resourceEffectiveElementWidth = 0;
  std::string resourcePackingLayout;
  std::string resourceUnpackIntent;
  std::int64_t vsetvlRegionCount = 0;
  std::string runtimeAVLSource;
  std::string runtimeABIOrder;
  std::string routeFamilyPlan;
  std::string providerSupportedMirror;
  std::string sourceRecordContract;
  std::string sourceSelectedVariant;
  std::string sourceSelectedInput;
  std::string sourceGeneratedFunction;
  std::string generatedArtifactIdentityContract;
  std::string generatedArtifactObjectPath;
  std::string generatedArtifactObjectSHA256;
  std::string generatedArtifactHeaderPath;
  std::string generatedArtifactHeaderSHA256;
  std::string measurementTarget;
  std::string measurementTargetProvenance;
  std::string measurementRuntimeCountSet;
  std::string measurementRuntimeCountProvenance;
  std::string pressureProfileLabel;
  std::string pressureProfileLabelProvenance;

  std::string primitiveChainContract;
  std::string primitiveChainKind;
  std::string primitiveContract;
  std::string primitiveKind;
  std::string wideningProductMultiplicandRoles;
  std::string wideningProductExtensionPolicy;
  std::string primitiveSourceLoad;
  std::string primitiveSourceExtension;
  std::string primitiveSourceDType;
  std::string primitiveSourceSignedness;
  std::int64_t primitiveSourceSEW = 0;
  std::string primitiveSourceLMUL;
  std::string primitiveProductDType;
  std::int64_t primitiveProductSEW = 0;
  std::string primitiveProductLMUL;
  std::string primitiveAccumulatorDType;
  std::int64_t primitiveAccumulatorSEW = 0;
  std::string primitiveAccumulatorLMUL;
  std::string primitiveResultDType;
  std::int64_t primitiveResultSEW = 0;
  std::string primitiveResultLMUL;
  std::string primitiveWideningProductRelation;
  std::string primitiveProductReductionChainRelation;
  std::string primitiveWideningProductIntrinsic;
  std::string primitiveReductionIntrinsic;
  std::string primitiveScalarSeedSplatIntrinsic;
  std::string primitiveAccumulatorLayout;
  std::string primitiveResultLayout;
  std::string primitiveReductionStoreVL;

  std::string scheduleDecisionContract;
  std::string scheduleDecision;
  std::string scheduleDecisionReason;
  std::string resourceCostContract;
  std::string resourceCostModel;
  std::int64_t resourceCostLoopBodySteps = 0;
  std::string resourceCostBlocker;
  std::string performanceAdmissionDecision;
  std::string performanceAdmissionClosure;
  std::string performanceAdmissionReopenRequirement;
  std::string beyondLocalRepairAdmissionContract;
  std::string beyondLocalRepairAdmissionDecision;
  std::string beyondLocalRepairAdmissionBlocker;
  std::string beyondLocalRepairAdmissionReopenRequirement;
  std::string realizationAdmissionContract;
  std::string realizationAdmissionDecision;
  std::string realizationAdmissionEvidence;
  std::string realizationAdmissionDispatchPolicy;
  std::string realizationAdmissionScheduleDecisionContract;
  std::string realizationAdmissionScheduleDecision;
  std::string realizationAdmissionScheduleDecisionReason;
  std::string targetProfile;
  std::string targetCapabilityProviderMirror;
  std::string targetCapabilityLegalityMirror;
  std::string measurementEvidenceID;
  std::string measurementClassification;
  std::string measurementOutcomeFamily;
  std::string measurementBestSpeedupRange;
  std::int64_t measurementSummaryRecordCount = 0;
  std::int64_t measurementRecordCount = 0;
  std::int64_t correctnessRecordCount = 0;

  std::string selectedDispatchCaseMirror;
  std::string selectedDispatchFallbackMirror;
  std::string selectedCaseVariant;
  std::string fallbackVariant;

  std::string dispatchPolicyPath;
  std::string dispatchPreference;
  bool routeSupportAllowed = false;
  bool correctnessExecutionAllowed = false;
  bool performanceSelectionAllowed = false;
  bool performanceWinClaimAllowed = false;
  bool correctnessFallbackPathSelected = false;
  bool performancePreferredPathSelected = false;
};

enum class RVVLowPrecisionRealizationAdmissionDecision {
  Realize,
  Defer,
  Deny,
};

struct RVVLowPrecisionSelectedBodyRealizationAdmission {
  std::string admissionContract;
  std::string admissionOwner;
  RVVLowPrecisionRealizationAdmissionDecision decision =
      RVVLowPrecisionRealizationAdmissionDecision::Deny;
  std::string selectedCandidateID;
  std::string pressureProfileContract;
  std::string measurementEvidenceID;
  std::string dispatchPolicyPath;
  std::string scheduleDecisionContract;
  std::string scheduleDecision;
  std::string scheduleDecisionReason;
  std::string diagnostic;

  bool admitsRealization() const;
};

RVVLowPrecisionSameTargetMeasurementRecord
buildRVVPackedI4Gate4SameTargetMeasurementRecord(
    const RVVLowPrecisionContractionResourceSelection &selection);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord>
buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceInput(
    const llvm::json::Object &evidenceInput, llvm::StringRef context);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord>
buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceRoot(
    const llvm::json::Object &evidenceRoot, llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformanceMeasurementOutcome>
buildRVVLowPrecisionPerformanceMeasurementOutcomeFromSameTargetRecord(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

RVVLowPrecisionSameTargetMeasurementPolicyInput
buildRVVLowPrecisionSameTargetMeasurementPolicyInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceInput, llvm::StringRef context);

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceRoot(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot, llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformanceMeasurementOutcome>
consumeRVVLowPrecisionSameTargetMeasurementPolicyInput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionProductionPressureProfile>
buildRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionProductionPressureProfile>
buildRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error populateRVVLowPrecisionSelectedDispatchPolicyOutput(
    const RVVLowPrecisionContractionResourceSelection &selection,
    RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error populateRVVLowPrecisionSelectedBodyRealizationAdmissionProof(
    RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::StringRef stringifyRVVLowPrecisionRealizationAdmissionDecision(
    RVVLowPrecisionRealizationAdmissionDecision decision);

llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission>
admitRVVLowPrecisionSelectedBodyRealization(
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission>
admitRVVLowPrecisionSelectedBodyRealization(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionSelectedBodyRealizationAdmission(
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionSelectedBodyRealizationAdmission(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context);

llvm::StringRef stringifyRVVLowPrecisionPerformanceMeasurementDiagnosisKind(
    RVVLowPrecisionPerformanceMeasurementDiagnosisKind kind);

RVVLowPrecisionPerformancePolicyHandoff
diagnoseRVVLowPrecisionPerformancePolicyHandoff(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyHandoff
diagnoseRVVLowPrecisionPerformancePolicyHandoff(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyHandoff
diagnoseRVVLowPrecisionPerformancePolicyHandoff(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

RVVLowPrecisionPerformancePolicyDecision
resolveRVVLowPrecisionDispatchPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionPerformanceMeasurementOutcome &outcome,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVLOWPRECISIONPERFORMANCEPOLICY_H
