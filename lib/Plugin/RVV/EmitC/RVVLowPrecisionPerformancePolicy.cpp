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
    "no-win");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementOutcomeFamily("no-win");
constexpr llvm::StringLiteral kPackedI4Gate4MeasurementBestSpeedupRange(
    "0.895307..1.027027");
constexpr llvm::StringLiteral
    kPackedI4DequantClampGate4MeasurementBestSpeedupRange(
        "0.874735..1.061579");
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
constexpr llvm::StringLiteral kProductionPressureProfileContract(
    "rvv-low-precision-production-pressure-profile.v1");
constexpr llvm::StringLiteral kProductionPressureProfileAuthority(
    "selected-typed-rvv-provider-resource-facts-plus-source-backed-"
    "same-target-measurement-and-selected-dispatch-policy-inputs");
constexpr llvm::StringLiteral kSelectedBodyRealizationAdmissionContract(
    "rvv-low-precision-selected-body-realization-admission.v1");
constexpr llvm::StringLiteral kSelectedBodyRealizationAdmissionOwner(
    "rvv-contraction-selected-body-realization-owner");
constexpr llvm::StringLiteral kSourceBackedMeasurementRecordContract(
    "rvv-low-precision-source-backed-artifact-measurement-record.v1");
constexpr llvm::StringLiteral kGeneratedArtifactIdentityContract(
    "generated-object-header-sha256-after-target-artifact-validation.v1");
constexpr llvm::StringLiteral kMeasurementTargetProvenance(
    "same-target-measurement-workflow-ssh-target.v1");
constexpr llvm::StringLiteral kMeasurementRuntimeCountProvenance(
    "same-target-measurement-config-input-sizes.v1");
constexpr llvm::StringLiteral kProductionPressureProfileLabel(
    "low-precision-quantized-contraction-production-pressure");
constexpr llvm::StringLiteral kProductionPressureProfileLabelProvenance(
    "non-authoritative-pressure-label-derived-from-selected-typed-rvv-"
    "provider-facts-and-source-backed-measurement-record");
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
constexpr std::int64_t
    kPackedI4DequantClampGate4MeasurementSummaryRecordCount = 24;
constexpr std::int64_t kPackedI4DequantClampGate4MeasurementRecordCount = 120;
constexpr std::int64_t kPackedI4DequantClampGate4CorrectnessRecordCount = 24;

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

bool containsPolicyMarker(llvm::StringRef value, llvm::StringRef marker) {
  std::string lowerStorage = value.lower();
  llvm::StringRef lower(lowerStorage);
  return lower.contains(marker);
}

bool containsLabelOnlyPressureMarker(llvm::StringRef value) {
  return containsPolicyMarker(value, "label-only") ||
         containsPolicyMarker(value, "q8") ||
         containsPolicyMarker(value, "q4") ||
         containsPolicyMarker(value, "llama");
}

llvm::StringRef
getPackedI4SourceSelectedVariantForCandidate(llvm::StringRef candidateID);

llvm::StringRef
getPackedI4SourceGeneratedFunctionForCandidate(llvm::StringRef candidateID);

llvm::Error rejectLabelOnlyPressureMarker(llvm::StringRef context,
                                          llvm::StringRef label,
                                          llvm::StringRef value) {
  if (!containsLabelOnlyPressureMarker(value))
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " pressure-profile boundary rejects label-only q8/q4 pressure marker in " +
      label + " '" + value +
      "'; typed tcrv_rvv/provider facts and source-backed measurement "
      "tie-backs are required");
}

llvm::Error rejectMetadataOnlyPressureMarker(llvm::StringRef context,
                                             llvm::StringRef label,
                                             llvm::StringRef value) {
  if (!containsPolicyMarker(value, "metadata-only"))
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " pressure-profile boundary rejects metadata-only pressure fact in " +
      label + " '" + value +
      "'; provider-owned facts must match the same-target measurement and "
      "selected-dispatch policy inputs");
}

llvm::Error rejectNoWinDispatchPerformancePreferenceMarker(
    llvm::StringRef context, llvm::StringRef label, llvm::StringRef value) {
  if (!containsPolicyMarker(value, "performance-preferred"))
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " selected-dispatch no-win policy boundary rejects performance-"
      "preferred marker in " +
      label + " '" + value +
      "'; route support and correctness execution do not authorize "
      "performance-preferred dispatch without measured-win evidence");
}

llvm::Error requireSelectedDispatchMirrorToken(llvm::StringRef context,
                                               llvm::StringRef mirrorLabel,
                                               llvm::StringRef mirror,
                                               llvm::StringRef factLabel,
                                               const llvm::Twine &token) {
  std::string tokenStorage = token.str();
  if (mirror.contains(tokenStorage))
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires " + mirrorLabel + " to mirror " +
      factLabel + " token '" + tokenStorage + "' but found '" + mirror + "'");
}

llvm::Error verifyRVVLowPrecisionSelectedDispatchMirrorTieBack(
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch case mirror",
          dispatchBoundary.selectedDispatchCaseMirror,
          "selected-dispatch case variant",
          llvm::Twine("selected_dispatch_case_mirror:@") +
              dispatchBoundary.selectedCaseVariant + ";"))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch case mirror",
          dispatchBoundary.selectedDispatchCaseMirror, "selected case role",
          llvm::Twine("role=") + dispatchBoundary.selectedCaseRole))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch case mirror",
          dispatchBoundary.selectedDispatchCaseMirror, "runtime guard flag",
          llvm::Twine("runtime_guard_required=") +
              (dispatchBoundary.runtimeGuardRequired ? "true" : "false")))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "selected dispatch runtime guard mirror fact",
          dispatchBoundary.runtimeGuard))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch case mirror",
          dispatchBoundary.selectedDispatchCaseMirror, "runtime guard",
          llvm::Twine("runtime_guard=") + dispatchBoundary.runtimeGuard))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch case mirror",
          dispatchBoundary.selectedDispatchCaseMirror, "selected case origin",
          llvm::Twine("origin=") + dispatchBoundary.selectedCaseOrigin))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch case mirror",
          dispatchBoundary.selectedDispatchCaseMirror, "selected case policy",
          llvm::Twine("policy=") + dispatchBoundary.selectedCasePolicy))
    return error;

  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch fallback mirror",
          dispatchBoundary.selectedDispatchFallbackMirror, "fallback variant",
          llvm::Twine("selected_dispatch_fallback_mirror:@") +
              dispatchBoundary.fallbackVariant + ";"))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch fallback mirror",
          dispatchBoundary.selectedDispatchFallbackMirror,
          "fallback path role",
          llvm::Twine("role=") + dispatchBoundary.fallbackPathRole))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch fallback mirror",
          dispatchBoundary.selectedDispatchFallbackMirror, "fallback role",
          llvm::Twine("fallback_role=") + dispatchBoundary.fallbackRole))
    return error;
  if (llvm::Error error = requireSelectedDispatchMirrorToken(
          context, "selected dispatch fallback mirror",
          dispatchBoundary.selectedDispatchFallbackMirror, "fallback origin",
          llvm::Twine("origin=") + dispatchBoundary.fallbackOrigin))
    return error;
  return requireSelectedDispatchMirrorToken(
      context, "selected dispatch fallback mirror",
      dispatchBoundary.selectedDispatchFallbackMirror, "fallback policy",
      llvm::Twine("policy=") + dispatchBoundary.fallbackPolicy);
}

llvm::Error rejectPressureProfileMarkerOnlyFacts(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  struct Field {
    llvm::StringRef label;
    llvm::StringRef value;
  };
  Field fields[] = {
      {"selection candidate", selection.selectedCandidateID},
      {"selection reason", selection.selectionReason},
      {"measurement authority", input.authority},
      {"measurement evidence id", input.measurementEvidenceID},
      {"measurement provider candidate",
       input.providerResourceSelectedCandidate},
      {"provider route-family plan", input.providerResourceRouteFamilyPlan},
      {"provider-supported mirror", input.providerSupportedMirror},
      {"selected-dispatch case variant", dispatchBoundary.selectedCaseVariant},
      {"selected-dispatch case policy", dispatchBoundary.selectedCasePolicy},
      {"selected-dispatch case mirror",
       dispatchBoundary.selectedDispatchCaseMirror},
      {"selected-dispatch fallback mirror",
       dispatchBoundary.selectedDispatchFallbackMirror},
  };
  for (const Field &field : fields) {
    if (llvm::Error error =
            rejectLabelOnlyPressureMarker(context, field.label, field.value))
      return error;
    if (llvm::Error error =
            rejectMetadataOnlyPressureMarker(context, field.label, field.value))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error rejectSourceBackedRecordMarkerOnlyFacts(
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  struct Field {
    llvm::StringRef label;
    llvm::StringRef value;
  };
  Field fields[] = {
      {"source selected variant", input.sourceSelectedVariant},
      {"source selected input", input.sourceSelectedInput},
      {"source generated function", input.sourceGeneratedFunction},
      {"generated artifact object path", input.generatedArtifactObjectPath},
      {"generated artifact object sha256",
       input.generatedArtifactObjectSHA256},
      {"generated artifact header path", input.generatedArtifactHeaderPath},
      {"generated artifact header sha256",
       input.generatedArtifactHeaderSHA256},
      {"measurement target", input.measurementTarget},
      {"measurement runtime count set", input.measurementRuntimeCountSet},
      {"pressure profile label", input.pressureProfileLabel},
      {"pressure profile label provenance",
       input.pressureProfileLabelProvenance},
  };
  for (const Field &field : fields) {
    if (llvm::Error error =
            rejectLabelOnlyPressureMarker(context, field.label, field.value))
      return error;
    if (llvm::Error error =
            rejectMetadataOnlyPressureMarker(context, field.label, field.value))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error verifySourceBackedMeasurementRecordFacts(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    llvm::StringRef context) {
  if (llvm::Error error =
          requirePolicyString(context, "source record contract",
                              input.sourceRecordContract,
                              kSourceBackedMeasurementRecordContract))
    return error;
  if (llvm::Error error =
          rejectSourceBackedRecordMarkerOnlyFacts(input, context))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "source selected variant", input.sourceSelectedVariant))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "source selected variant", input.sourceSelectedVariant,
          getPackedI4SourceSelectedVariantForCandidate(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "source selected input", input.sourceSelectedInput))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "source generated function", input.sourceGeneratedFunction))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "source generated function", input.sourceGeneratedFunction,
          getPackedI4SourceGeneratedFunctionForCandidate(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "generated artifact identity contract",
                              input.generatedArtifactIdentityContract,
                              kGeneratedArtifactIdentityContract))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "generated artifact object path",
          input.generatedArtifactObjectPath))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "generated artifact object sha256",
          input.generatedArtifactObjectSHA256))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "generated artifact header path",
          input.generatedArtifactHeaderPath))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "generated artifact header sha256",
          input.generatedArtifactHeaderSHA256))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement target",
                              input.measurementTarget, input.targetProfile))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement target provenance",
                              input.measurementTargetProvenance,
                              kMeasurementTargetProvenance))
    return error;
  if (llvm::Error error = requireNonEmptyPolicyString(
          context, "measurement runtime count set",
          input.measurementRuntimeCountSet))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement runtime count provenance",
                              input.measurementRuntimeCountProvenance,
                              kMeasurementRuntimeCountProvenance))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "pressure profile label",
                              input.pressureProfileLabel,
                              kProductionPressureProfileLabel))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "pressure profile label provenance",
                              input.pressureProfileLabelProvenance,
                              kProductionPressureProfileLabelProvenance))
    return error;
  return llvm::Error::success();
}

bool hasPackedI4SiblingRouteMeasurement(
    const RVVLowPrecisionContractionResourceSelection &selection,
    llvm::StringRef providerSelectedCandidate) {
  return selection.hasSelection &&
         isRVVLowPrecisionResourcePackedI4CandidateID(
             selection.selectedCandidateID) &&
         isRVVLowPrecisionResourcePackedI4CandidateID(
             providerSelectedCandidate) &&
         providerSelectedCandidate != selection.selectedCandidateID;
}

llvm::StringRef
getPackedI4Gate4MeasurementBestSpeedupRangeForCandidate(
    llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return kPackedI4DequantClampGate4MeasurementBestSpeedupRange;
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return kPackedI4Gate4MeasurementBestSpeedupRange;
  return {};
}

std::int64_t getPackedI4Gate4MeasurementSummaryRecordCountForCandidate(
    llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return kPackedI4DequantClampGate4MeasurementSummaryRecordCount;
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return kPackedI4Gate4MeasurementSummaryRecordCount;
  return 0;
}

std::int64_t
getPackedI4Gate4MeasurementRecordCountForCandidate(llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return kPackedI4DequantClampGate4MeasurementRecordCount;
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return kPackedI4Gate4MeasurementRecordCount;
  return 0;
}

std::int64_t getPackedI4Gate4CorrectnessRecordCountForCandidate(
    llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return kPackedI4DequantClampGate4CorrectnessRecordCount;
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return kPackedI4Gate4CorrectnessRecordCount;
  return 0;
}

bool isPackedI4DequantClampCandidate(llvm::StringRef candidateID) {
  return candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate;
}

llvm::StringRef
getPackedI4SourceSelectedVariantForCandidate(llvm::StringRef candidateID) {
  return isPackedI4DequantClampCandidate(candidateID)
             ? llvm::StringRef("pre_realized_body_rvv_product_reduce_dequant_"
                               "clamp")
             : llvm::StringRef("pre_realized_body_rvv_product_reduce_"
                               "dequantize");
}

llvm::StringRef
getPackedI4SourceSelectedInputForCandidate(llvm::StringRef candidateID) {
  return isPackedI4DequantClampCandidate(candidateID)
             ? llvm::StringRef(
                   "test/Target/RVV/pre-realized-selected-body-artifact-"
                   "widening-product-reduce-dequant-clamp-f32-packed-i4.mlir")
             : llvm::StringRef(
                   "test/Target/RVV/pre-realized-selected-body-artifact-"
                   "widening-product-reduce-dequantize-f32-packed-i4.mlir");
}

llvm::StringRef
getPackedI4SourceGeneratedFunctionForCandidate(llvm::StringRef candidateID) {
  return isPackedI4DequantClampCandidate(candidateID)
             ? llvm::StringRef(
                   "tcrv_emitc_pre_realized_body_product_reduce_dequant_clamp_"
                   "kernel_pre_realized_body_rvv_product_reduce_dequant_clamp")
             : llvm::StringRef(
                   "tcrv_emitc_pre_realized_body_product_reduce_dequantize_"
                   "kernel_pre_realized_body_rvv_product_reduce_dequantize");
}

llvm::StringRef getPackedI4OpKindForCandidate(llvm::StringRef candidateID) {
  if (candidateID == kRVVLowPrecisionResourceDequantClampPackedI4Candidate)
    return "widening_product_reduce_dequant_clamp_f32";
  if (candidateID == kRVVLowPrecisionResourceDequantPackedI4Candidate)
    return "widening_product_reduce_dequantize_f32";
  return {};
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

llvm::Expected<const llvm::json::Object *>
requireEvidenceRootObject(const llvm::json::Object &evidenceRoot,
                          llvm::StringRef context, llvm::StringRef key) {
  if (const llvm::json::Object *value = evidenceRoot.getObject(key))
    return value;
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement evidence root object field '" + key +
      "'");
}

llvm::Expected<std::string>
requireEvidenceRootString(const llvm::json::Object &evidenceRoot,
                          llvm::StringRef context, llvm::StringRef key) {
  if (std::optional<llvm::StringRef> value = evidenceRoot.getString(key))
    return value->str();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement evidence root string field '" + key +
      "'");
}

llvm::Expected<bool>
requireEvidenceRootBool(const llvm::json::Object &evidenceRoot,
                        llvm::StringRef context, llvm::StringRef key) {
  if (std::optional<bool> value = evidenceRoot.getBoolean(key))
    return *value;
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement evidence root boolean field '" + key +
      "'");
}

llvm::Error requireEvidenceRootString(llvm::StringRef context,
                                      llvm::StringRef label,
                                      llvm::StringRef actual,
                                      llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires same-target measurement evidence " +
      label + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireEvidenceRootInt(llvm::StringRef context,
                                   llvm::StringRef label, std::int64_t actual,
                                   std::int64_t expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires same-target measurement evidence " +
      label + " " + llvm::Twine(expected) + " but found " +
      llvm::Twine(actual));
}

llvm::Error requireEvidenceRootBool(llvm::StringRef context,
                                    llvm::StringRef label, bool actual,
                                    bool expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) + " requires same-target measurement evidence " +
      label + " " + (expected ? "true" : "false") + " but found " +
      (actual ? "true" : "false"));
}

llvm::Error requireEvidenceRootStringField(
    const llvm::json::Object &object, llvm::StringRef context,
    llvm::StringRef objectLabel, llvm::StringRef key,
    llvm::StringRef expected) {
  const std::string fieldLabel = (llvm::Twine(objectLabel) + "." + key).str();
  if (std::optional<llvm::StringRef> actual = object.getString(key))
    return requireEvidenceRootString(context, fieldLabel, *actual, expected);
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement evidence root string field '" +
      fieldLabel + "'");
}

llvm::Error requireEvidenceRootIntField(const llvm::json::Object &object,
                                        llvm::StringRef context,
                                        llvm::StringRef objectLabel,
                                        llvm::StringRef key,
                                        std::int64_t expected) {
  const std::string fieldLabel = (llvm::Twine(objectLabel) + "." + key).str();
  if (std::optional<std::int64_t> actual = object.getInteger(key))
    return requireEvidenceRootInt(context, fieldLabel, *actual, expected);
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement evidence root integer field '" +
      fieldLabel + "'");
}

llvm::Error requireEvidenceRootBoolField(const llvm::json::Object &object,
                                         llvm::StringRef context,
                                         llvm::StringRef objectLabel,
                                         llvm::StringRef key, bool expected) {
  const std::string fieldLabel = (llvm::Twine(objectLabel) + "." + key).str();
  if (std::optional<bool> actual = object.getBoolean(key))
    return requireEvidenceRootBool(context, fieldLabel, *actual, expected);
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " requires same-target measurement evidence root boolean field '" +
      fieldLabel + "'");
}

llvm::Error verifyPackedI4SameTargetEvidenceRoot(
    const llvm::json::Object &evidenceRoot,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    llvm::StringRef context) {
  llvm::StringRef expectedOpKind =
      getPackedI4OpKindForCandidate(record.providerResourceSelectedCandidate);
  llvm::StringRef expectedBaseline =
      getRVVLowPrecisionResourcePackedI4PerformanceBaselineForCandidate(
          record.providerResourceSelectedCandidate);
  if (expectedOpKind.empty() || expectedBaseline.empty())
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires a known packed-i4 provider candidate before consuming a "
        "same-target measurement evidence root");

  llvm::Expected<std::string> status =
      requireEvidenceRootString(evidenceRoot, context, "status");
  if (!status)
    return status.takeError();
  if (llvm::Error error =
          requireEvidenceRootString(context, "root status", *status, "success"))
    return error;
  llvm::Expected<std::string> opKind =
      requireEvidenceRootString(evidenceRoot, context, "op_kind");
  if (!opKind)
    return opKind.takeError();
  if (llvm::Error error = requireEvidenceRootString(context, "root op_kind",
                                                    *opKind, expectedOpKind))
    return error;
  llvm::Expected<std::string> baseline =
      requireEvidenceRootString(evidenceRoot, context, "baseline_identity");
  if (!baseline)
    return baseline.takeError();
  if (llvm::Error error = requireEvidenceRootString(
          context, "root baseline_identity", *baseline, expectedBaseline))
    return error;
  llvm::Expected<std::string> sshTarget =
      requireEvidenceRootString(evidenceRoot, context, "ssh_target");
  if (!sshTarget)
    return sshTarget.takeError();
  if (llvm::Error error =
          requireEvidenceRootString(context, "root ssh_target", *sshTarget,
                                    "rvv"))
    return error;
  llvm::Expected<std::string> timingMethod =
      requireEvidenceRootString(evidenceRoot, context, "timing_method");
  if (!timingMethod)
    return timingMethod.takeError();
  if (llvm::Error error =
          requireEvidenceRootString(context, "root timing_method",
                                    *timingMethod,
                                    "clock_gettime(CLOCK_MONOTONIC_RAW)"))
    return error;
  llvm::Expected<std::string> inputMode =
      requireEvidenceRootString(evidenceRoot, context, "input_mode");
  if (!inputMode)
    return inputMode.takeError();
  if (llvm::Error error = requireEvidenceRootString(
          context, "root input_mode", *inputMode, "pre-realized-selected-body"))
    return error;
  llvm::Expected<bool> dryRun =
      requireEvidenceRootBool(evidenceRoot, context, "dry_run");
  if (!dryRun)
    return dryRun.takeError();
  if (llvm::Error error =
          requireEvidenceRootBool(context, "root dry_run", *dryRun, false))
    return error;
  llvm::Expected<bool> sshEvidence =
      requireEvidenceRootBool(evidenceRoot, context, "ssh_evidence");
  if (!sshEvidence)
    return sshEvidence.takeError();
  if (llvm::Error error = requireEvidenceRootBool(context, "root ssh_evidence",
                                                  *sshEvidence, true))
    return error;
  llvm::Expected<bool> packedMetadataSelected =
      requireEvidenceRootBool(evidenceRoot, context,
                              "packed_i4_resource_metadata_selected");
  if (!packedMetadataSelected)
    return packedMetadataSelected.takeError();
  if (llvm::Error error = requireEvidenceRootBool(
          context, "root packed_i4_resource_metadata_selected",
          *packedMetadataSelected, true))
    return error;

  llvm::Expected<const llvm::json::Object *> resultClassification =
      requireEvidenceRootObject(evidenceRoot, context, "result_classification");
  if (!resultClassification)
    return resultClassification.takeError();
  if (llvm::Error error = requireEvidenceRootStringField(
          **resultClassification, context, "result_classification",
          "classification", record.measurementClassification))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **resultClassification, context, "result_classification",
          "outcome_family", record.measurementOutcomeFamily))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **resultClassification, context, "result_classification",
          "best_speedup_range", record.measurementBestSpeedupRange))
    return error;
  if (llvm::Error error = requireEvidenceRootIntField(
          **resultClassification, context, "result_classification",
          "summary_record_count", record.measurementSummaryRecordCount))
    return error;
  if (llvm::Error error = requireEvidenceRootIntField(
          **resultClassification, context, "result_classification",
          "measurement_record_count", record.measurementRecordCount))
    return error;
  if (llvm::Error error = requireEvidenceRootIntField(
          **resultClassification, context, "result_classification",
          "correctness_record_count", record.correctnessRecordCount))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **resultClassification, context, "result_classification",
          "timing_method", *timingMethod))
    return error;
  if (llvm::Error error = requireEvidenceRootBoolField(
          **resultClassification, context, "result_classification",
          "correctness_before_timing", true))
    return error;

  llvm::Expected<const llvm::json::Object *> measurementHarness =
      requireEvidenceRootObject(evidenceRoot, context, "measurement_harness");
  if (!measurementHarness)
    return measurementHarness.takeError();
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "baseline_identity", expectedBaseline))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "generated_function", record.sourceGeneratedFunction))
    return error;
  if (llvm::Error error = requireEvidenceRootBoolField(
          **measurementHarness, context, "measurement_harness",
          "correctness_before_timing", true))
    return error;
  if (llvm::Error error = requireEvidenceRootBoolField(
          **measurementHarness, context, "measurement_harness",
          "packed_i4_reference_oracle", true))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_schedule_decision_contract",
          record.providerScheduleDecisionContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_schedule_decision", record.providerScheduleDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_schedule_decision_reason",
          record.providerScheduleDecisionReason))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_resource_cost_contract",
          record.providerResourceCostContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_resource_cost_model", record.providerResourceCostModel))
    return error;
  if (llvm::Error error = requireEvidenceRootIntField(
          **measurementHarness, context, "measurement_harness",
          "provider_resource_cost_loop_body_steps",
          record.providerResourceCostLoopBodySteps))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_resource_cost_blocker",
          record.providerResourceCostBlocker))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_performance_admission_decision",
          record.providerPerformanceAdmissionDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_beyond_local_repair_admission_contract",
          record.providerBeyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_beyond_local_repair_admission_decision",
          record.providerBeyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_beyond_local_repair_admission_blocker",
          record.providerBeyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_beyond_local_repair_admission_reopen_requirement",
          record.providerBeyondLocalRepairAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_realization_admission_schedule_decision_contract",
          record.providerRealizationAdmissionScheduleDecisionContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_realization_admission_schedule_decision",
          record.providerRealizationAdmissionScheduleDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **measurementHarness, context, "measurement_harness",
          "provider_realization_admission_schedule_decision_reason",
          record.providerRealizationAdmissionScheduleDecisionReason))
    return error;

  llvm::Expected<const llvm::json::Object *> scheduleEvidence =
      requireEvidenceRootObject(evidenceRoot, context,
                                "measurement_schedule_decision_evidence");
  if (!scheduleEvidence)
    return scheduleEvidence.takeError();
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_schedule_decision_contract",
          record.providerScheduleDecisionContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_schedule_decision", record.providerScheduleDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_schedule_decision_reason",
          record.providerScheduleDecisionReason))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_resource_cost_contract",
          record.providerResourceCostContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_resource_cost_model", record.providerResourceCostModel))
    return error;
  if (llvm::Error error = requireEvidenceRootIntField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_resource_cost_loop_body_steps",
          record.providerResourceCostLoopBodySteps))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_resource_cost_blocker",
          record.providerResourceCostBlocker))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_performance_admission_decision",
          record.providerPerformanceAdmissionDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_beyond_local_repair_admission_contract",
          record.providerBeyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_beyond_local_repair_admission_decision",
          record.providerBeyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_beyond_local_repair_admission_blocker",
          record.providerBeyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_beyond_local_repair_admission_reopen_requirement",
          record.providerBeyondLocalRepairAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_realization_admission_schedule_decision_contract",
          record.providerRealizationAdmissionScheduleDecisionContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_realization_admission_schedule_decision",
          record.providerRealizationAdmissionScheduleDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **scheduleEvidence, context, "measurement_schedule_decision_evidence",
          "provider_realization_admission_schedule_decision_reason",
          record.providerRealizationAdmissionScheduleDecisionReason))
    return error;

  llvm::Expected<const llvm::json::Object *> packedOracle =
      requireEvidenceRootObject(evidenceRoot, context,
                                "packed_i4_reference_oracle");
  if (!packedOracle)
    return packedOracle.takeError();
  if (llvm::Error error = requireEvidenceRootStringField(
          **packedOracle, context, "packed_i4_reference_oracle",
          "baseline_identity", expectedBaseline))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **packedOracle, context, "packed_i4_reference_oracle",
          "operand_form", record.providerResourceOperandForm))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **packedOracle, context, "packed_i4_reference_oracle",
          "packing_layout", record.providerResourcePackingLayout))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **packedOracle, context, "packed_i4_reference_oracle",
          "unpack_intent", record.providerResourceUnpackIntent))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **packedOracle, context, "packed_i4_reference_oracle",
          "provider_schedule_decision_contract",
          record.providerScheduleDecisionContract))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **packedOracle, context, "packed_i4_reference_oracle",
          "provider_schedule_decision", record.providerScheduleDecision))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **packedOracle, context, "packed_i4_reference_oracle",
          "provider_schedule_decision_reason",
          record.providerScheduleDecisionReason))
    return error;

  llvm::Expected<const llvm::json::Object *> maturityInput =
      requireEvidenceRootObject(evidenceRoot, context,
                                "performance_maturity_contract_evidence_input");
  if (!maturityInput)
    return maturityInput.takeError();
  if (llvm::Error error = requireEvidenceRootStringField(
          **maturityInput, context,
          "performance_maturity_contract_evidence_input",
          "measurement_evidence_id", record.measurementEvidenceID))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **maturityInput, context,
          "performance_maturity_contract_evidence_input",
          "contract_alignment", "matches-provider-maturity-outcome"))
    return error;

  llvm::Expected<const llvm::json::Object *> providerTieBack =
      requireEvidenceRootObject(evidenceRoot, context, "provider_feedback_tie_back");
  if (!providerTieBack)
    return providerTieBack.takeError();
  if (llvm::Error error = requireEvidenceRootStringField(
          **providerTieBack, context, "provider_feedback_tie_back",
          "baseline_identity", expectedBaseline))
    return error;
  if (llvm::Error error = requireEvidenceRootBoolField(
          **providerTieBack, context, "provider_feedback_tie_back",
          "packed_i4_resource_metadata_selected", true))
    return error;
  if (llvm::Error error = requireEvidenceRootBoolField(
          **providerTieBack, context, "provider_feedback_tie_back",
          "performance_win_claim_allowed", false))
    return error;
  if (llvm::Error error = requireEvidenceRootStringField(
          **providerTieBack, context, "provider_feedback_tie_back",
          "result_alignment", "matches-provider-maturity-outcome"))
    return error;

  return llvm::Error::success();
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
          getRVVLowPrecisionResourceSelectionReasonForCandidate(
              selection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 planning contract", selection.planningContract,
          kRVVLowPrecisionResourcePlanningContract))
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
          context, "packed-i4 vsetvl region count",
          selection.vsetvlRegionCount,
          kRVVLowPrecisionResourcePackedI4VSetVLRegions))
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
          context, "packed-i4 runtime AVL source", selection.runtimeAVLSource,
          kRVVGearboxRuntimeAVLSourceN))
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
          context, "widening product multiplicand roles",
          selection.wideningProductMultiplicandRoleSummary,
          kRVVLowPrecisionResourceWideningProductMultiplicandRoles))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "widening product extension policy",
          selection.wideningProductExtensionPolicy,
          kRVVLowPrecisionResourceWideningProductExtensionPolicy))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive source load", selection.primitiveSourceLoadKind,
          kRVVLowPrecisionResourcePrimitiveSourceLoad))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "primitive source extension",
          selection.primitiveSourceExtensionKind,
          kRVVLowPrecisionResourcePrimitiveSourceExtension))
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
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 resource cost contract",
          selection.resourceCostContract,
          kRVVLowPrecisionResourcePackedI4CostContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 resource cost model",
          selection.resourceCostModel, kRVVLowPrecisionResourcePackedI4CostModel))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "packed-i4 resource cost loop-body steps",
          selection.resourceCostLoopBodySteps,
          kRVVLowPrecisionResourcePackedI4CostLoopBodySteps))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 resource cost blocker",
          selection.resourceCostBlocker,
          kRVVLowPrecisionResourcePackedI4CostBlocker))
    return error;
  if (selection.performanceAdmissionDecision !=
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision &&
      selection.performanceAdmissionDecision !=
          kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionDecision)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider performance admission decision to be a consumed "
        "resource-cost decision: expected '" +
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision +
        "' or '" +
        kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionDecision +
        "' but found '" + selection.performanceAdmissionDecision + "'");
  if (selection.performanceAdmissionClosure !=
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure &&
      selection.performanceAdmissionClosure !=
          kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionClosure)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider performance admission closure to be a consumed "
        "no-safe-repair or measured-win closure: expected '" +
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure + "' or '" +
        kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionClosure +
        "' but found '" + selection.performanceAdmissionClosure + "'");
  if (selection.performanceAdmissionReopenRequirement !=
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement &&
      selection.performanceAdmissionReopenRequirement !=
          kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionReopenRequirement)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider performance admission reopen requirement to be a "
        "consumed no-safe-repair or measured-win requirement: expected '" +
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement +
        "' or '" +
        kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionReopenRequirement +
        "' but found '" +
        selection.performanceAdmissionReopenRequirement + "'");
  if (llvm::Error error = requirePolicyString(
          context, "provider beyond-local repair admission contract",
          selection.beyondLocalRepairAdmissionContract,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract))
    return error;
  if (selection.beyondLocalRepairAdmissionDecision !=
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision &&
      selection.beyondLocalRepairAdmissionDecision !=
          kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionDecision)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider beyond-local repair admission decision to be a "
        "consumed campaign-level decision: expected '" +
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision +
        "' or '" +
        kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionDecision +
        "' but found '" + selection.beyondLocalRepairAdmissionDecision + "'");
  if (selection.beyondLocalRepairAdmissionBlocker !=
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker &&
      selection.beyondLocalRepairAdmissionBlocker !=
          kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionBlocker)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider beyond-local repair admission blocker to be the "
        "current no-further-repair blocker or measured-win none marker: "
        "expected '" +
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker +
        "' or '" +
        kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionBlocker +
        "' but found '" + selection.beyondLocalRepairAdmissionBlocker + "'");
  if (selection.beyondLocalRepairAdmissionReopenRequirement !=
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement &&
      selection.beyondLocalRepairAdmissionReopenRequirement !=
          kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionReopenRequirement)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider beyond-local repair admission reopen requirement "
        "to be the current typed/provider repair requirement or measured-win "
        "none marker: expected '" +
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement +
        "' or '" +
        kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionReopenRequirement +
        "' but found '" +
        selection.beyondLocalRepairAdmissionReopenRequirement + "'");
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
  if (llvm::Error error =
          verifySourceBackedMeasurementRecordFacts(selection, input, context))
    return error;

  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider selected candidate",
          input.providerResourceSelectedCandidate, selection.selectedCandidateID))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider planning contract",
          input.providerResourcePlanningContract, selection.planningContract))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider operand form", input.providerResourceOperandForm,
          selection.operandForm))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider source signedness",
          input.providerResourceSourceSignedness, selection.sourceSignedness))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider storage element width",
          input.providerResourceStorageElementWidth,
          selection.storageElementWidth))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider effective element width",
          input.providerResourceEffectiveElementWidth,
          selection.effectiveElementWidth))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider packing layout",
          input.providerResourcePackingLayout, selection.packingLayout))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider unpack intent", input.providerResourceUnpackIntent,
          selection.unpackIntent))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider vsetvl region count",
          input.providerResourceVSetVLRegionCount,
          selection.vsetvlRegionCount))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider runtime AVL source",
          input.providerRuntimeAVLSource, selection.runtimeAVLSource))
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
          context, "provider resource cost contract",
          input.providerResourceCostContract,
          selection.resourceCostContract))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider resource cost model",
          input.providerResourceCostModel, selection.resourceCostModel))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider resource cost loop-body steps",
          input.providerResourceCostLoopBodySteps,
          selection.resourceCostLoopBodySteps))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider resource cost blocker",
          input.providerResourceCostBlocker, selection.resourceCostBlocker))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider performance admission decision",
          input.providerPerformanceAdmissionDecision,
          selection.performanceAdmissionDecision))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider performance admission closure",
          input.providerPerformanceAdmissionClosure,
          selection.performanceAdmissionClosure))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider performance admission reopen requirement",
          input.providerPerformanceAdmissionReopenRequirement,
          selection.performanceAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider beyond-local repair admission contract",
          input.providerBeyondLocalRepairAdmissionContract,
          selection.beyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider beyond-local repair admission decision",
          input.providerBeyondLocalRepairAdmissionDecision,
          selection.beyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider beyond-local repair admission blocker",
          input.providerBeyondLocalRepairAdmissionBlocker,
          selection.beyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context,
          "provider beyond-local repair admission reopen requirement",
          input.providerBeyondLocalRepairAdmissionReopenRequirement,
          selection.beyondLocalRepairAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider realization admission contract",
          input.providerRealizationAdmissionContract,
          selection.realizationAdmissionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission contract",
          input.providerRealizationAdmissionContract,
          kSelectedBodyRealizationAdmissionContract))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider realization admission decision",
          input.providerRealizationAdmissionDecision,
          selection.realizationAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission decision",
          input.providerRealizationAdmissionDecision,
          stringifyRVVLowPrecisionRealizationAdmissionDecision(
              RVVLowPrecisionRealizationAdmissionDecision::Realize)))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider realization admission evidence",
          input.providerRealizationAdmissionEvidence,
          selection.realizationAdmissionEvidence))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission evidence",
          input.providerRealizationAdmissionEvidence,
          input.measurementEvidenceID))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider realization admission dispatch policy",
          input.providerRealizationAdmissionDispatchPolicy,
          selection.realizationAdmissionDispatchPolicy))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context,
          "provider realization admission schedule decision contract",
          input.providerRealizationAdmissionScheduleDecisionContract,
          selection.realizationAdmissionScheduleDecisionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context,
          "provider realization admission schedule decision contract",
          input.providerRealizationAdmissionScheduleDecisionContract,
          input.providerScheduleDecisionContract))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider realization admission schedule decision",
          input.providerRealizationAdmissionScheduleDecision,
          selection.realizationAdmissionScheduleDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission schedule decision",
          input.providerRealizationAdmissionScheduleDecision,
          input.providerScheduleDecision))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context,
          "provider realization admission schedule decision reason",
          input.providerRealizationAdmissionScheduleDecisionReason,
          selection.realizationAdmissionScheduleDecisionReason))
    return error;
  if (llvm::Error error = requirePolicyString(
          context,
          "provider realization admission schedule decision reason",
          input.providerRealizationAdmissionScheduleDecisionReason,
          input.providerScheduleDecisionReason))
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
          context, "provider primitive contract",
          input.providerPrimitiveContract, selection.primitiveContractID))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive kind", input.providerPrimitiveKind,
          selection.primitiveKind))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider widening product multiplicand roles",
          input.providerWideningProductMultiplicandRoles,
          selection.wideningProductMultiplicandRoleSummary))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider widening product extension policy",
          input.providerWideningProductExtensionPolicy,
          selection.wideningProductExtensionPolicy))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive source load",
          input.providerPrimitiveSourceLoad,
          selection.primitiveSourceLoadKind))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive source extension",
          input.providerPrimitiveSourceExtension,
          selection.primitiveSourceExtensionKind))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive source dtype",
          input.providerPrimitiveSourceDType, selection.sourceElementTypeName))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive source signedness",
          input.providerPrimitiveSourceSignedness, selection.sourceSignedness))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider primitive source SEW",
          input.providerPrimitiveSourceSEW, selection.sourceSEW))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive source LMUL",
          input.providerPrimitiveSourceLMUL, selection.sourceLMUL))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive product dtype",
          input.providerPrimitiveProductDType, selection.productElementTypeName))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider primitive product SEW",
          input.providerPrimitiveProductSEW, selection.productSEW))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive product LMUL",
          input.providerPrimitiveProductLMUL, selection.productLMUL))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive accumulator dtype",
          input.providerPrimitiveAccumulatorDType,
          selection.accumulatorElementTypeName))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider primitive accumulator SEW",
          input.providerPrimitiveAccumulatorSEW, selection.accumulatorSEW))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive accumulator LMUL",
          input.providerPrimitiveAccumulatorLMUL, selection.accumulatorLMUL))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive result dtype",
          input.providerPrimitiveResultDType, selection.resultElementTypeName))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider primitive result SEW",
          input.providerPrimitiveResultSEW, selection.resultSEW))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive result LMUL",
          input.providerPrimitiveResultLMUL, selection.resultLMUL))
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
          context, "provider primitive widening-product intrinsic",
          input.providerPrimitiveWideningProductIntrinsic,
          selection.primitiveWideningProductIntrinsic))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive reduction intrinsic",
          input.providerPrimitiveReductionIntrinsic,
          selection.primitiveReductionIntrinsic))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive scalar seed splat intrinsic",
          input.providerPrimitiveScalarSeedSplatIntrinsic,
          selection.primitiveScalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive accumulator layout",
          input.providerPrimitiveAccumulatorLayout,
          selection.primitiveAccumulatorLayout))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive result layout",
          input.providerPrimitiveResultLayout, selection.primitiveResultLayout))
    return error;
  if (llvm::Error error = requireSameTargetPolicyInputTieBack(
          context, "provider primitive reduction store VL",
          input.providerPrimitiveReductionStoreVL,
          selection.primitiveReductionStoreVL))
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
  outcome.providerResourceCostContract = input.providerResourceCostContract;
  outcome.providerResourceCostModel = input.providerResourceCostModel;
  outcome.providerResourceCostLoopBodySteps =
      input.providerResourceCostLoopBodySteps;
  outcome.providerResourceCostBlocker = input.providerResourceCostBlocker;
  outcome.providerPerformanceAdmissionDecision =
      input.providerPerformanceAdmissionDecision;
  outcome.providerPerformanceAdmissionClosure =
      input.providerPerformanceAdmissionClosure;
  outcome.providerPerformanceAdmissionReopenRequirement =
      input.providerPerformanceAdmissionReopenRequirement;
  outcome.providerBeyondLocalRepairAdmissionContract =
      input.providerBeyondLocalRepairAdmissionContract;
  outcome.providerBeyondLocalRepairAdmissionDecision =
      input.providerBeyondLocalRepairAdmissionDecision;
  outcome.providerBeyondLocalRepairAdmissionBlocker =
      input.providerBeyondLocalRepairAdmissionBlocker;
  outcome.providerBeyondLocalRepairAdmissionReopenRequirement =
      input.providerBeyondLocalRepairAdmissionReopenRequirement;
  outcome.providerRealizationAdmissionContract =
      input.providerRealizationAdmissionContract;
  outcome.providerRealizationAdmissionDecision =
      input.providerRealizationAdmissionDecision;
  outcome.providerRealizationAdmissionEvidence =
      input.providerRealizationAdmissionEvidence;
  outcome.providerRealizationAdmissionDispatchPolicy =
      input.providerRealizationAdmissionDispatchPolicy;
  outcome.providerRealizationAdmissionScheduleDecisionContract =
      input.providerRealizationAdmissionScheduleDecisionContract;
  outcome.providerRealizationAdmissionScheduleDecision =
      input.providerRealizationAdmissionScheduleDecision;
  outcome.providerRealizationAdmissionScheduleDecisionReason =
      input.providerRealizationAdmissionScheduleDecisionReason;
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
  input.sourceRecordContract = record.sourceRecordContract;
  input.sourceSelectedVariant = record.sourceSelectedVariant;
  input.sourceSelectedInput = record.sourceSelectedInput;
  input.sourceGeneratedFunction = record.sourceGeneratedFunction;
  input.generatedArtifactIdentityContract =
      record.generatedArtifactIdentityContract;
  input.generatedArtifactObjectPath = record.generatedArtifactObjectPath;
  input.generatedArtifactObjectSHA256 = record.generatedArtifactObjectSHA256;
  input.generatedArtifactHeaderPath = record.generatedArtifactHeaderPath;
  input.generatedArtifactHeaderSHA256 = record.generatedArtifactHeaderSHA256;
  input.measurementTarget = record.measurementTarget;
  input.measurementTargetProvenance = record.measurementTargetProvenance;
  input.measurementRuntimeCountSet = record.measurementRuntimeCountSet;
  input.measurementRuntimeCountProvenance =
      record.measurementRuntimeCountProvenance;
  input.pressureProfileLabel = record.pressureProfileLabel;
  input.pressureProfileLabelProvenance =
      record.pressureProfileLabelProvenance;
  input.providerResourceSelectedCandidate =
      record.providerResourceSelectedCandidate;
  input.providerResourcePlanningContract =
      record.providerResourcePlanningContract;
  input.providerResourceOperandForm = record.providerResourceOperandForm;
  input.providerResourceSourceSignedness =
      record.providerResourceSourceSignedness;
  input.providerResourceStorageElementWidth =
      record.providerResourceStorageElementWidth;
  input.providerResourceEffectiveElementWidth =
      record.providerResourceEffectiveElementWidth;
  input.providerResourcePackingLayout =
      record.providerResourcePackingLayout;
  input.providerResourceUnpackIntent = record.providerResourceUnpackIntent;
  input.providerResourceVSetVLRegionCount =
      record.providerResourceVSetVLRegionCount;
  input.providerRuntimeAVLSource = record.providerRuntimeAVLSource;
  input.providerResourceRouteFamilyPlan =
      record.providerResourceRouteFamilyPlan;
  input.providerSupportedMirror = record.providerSupportedMirror;
  input.providerRuntimeABIOrder = record.providerRuntimeABIOrder;
  input.providerScheduleDecisionContract =
      record.providerScheduleDecisionContract;
  input.providerScheduleDecision = record.providerScheduleDecision;
  input.providerScheduleDecisionReason =
      record.providerScheduleDecisionReason;
  input.providerResourceCostContract = record.providerResourceCostContract;
  input.providerResourceCostModel = record.providerResourceCostModel;
  input.providerResourceCostLoopBodySteps =
      record.providerResourceCostLoopBodySteps;
  input.providerResourceCostBlocker = record.providerResourceCostBlocker;
  input.providerPerformanceAdmissionDecision =
      record.providerPerformanceAdmissionDecision;
  input.providerPerformanceAdmissionClosure =
      record.providerPerformanceAdmissionClosure;
  input.providerPerformanceAdmissionReopenRequirement =
      record.providerPerformanceAdmissionReopenRequirement;
  input.providerBeyondLocalRepairAdmissionContract =
      record.providerBeyondLocalRepairAdmissionContract;
  input.providerBeyondLocalRepairAdmissionDecision =
      record.providerBeyondLocalRepairAdmissionDecision;
  input.providerBeyondLocalRepairAdmissionBlocker =
      record.providerBeyondLocalRepairAdmissionBlocker;
  input.providerBeyondLocalRepairAdmissionReopenRequirement =
      record.providerBeyondLocalRepairAdmissionReopenRequirement;
  input.providerRealizationAdmissionContract =
      record.providerRealizationAdmissionContract;
  input.providerRealizationAdmissionDecision =
      record.providerRealizationAdmissionDecision;
  input.providerRealizationAdmissionEvidence =
      record.providerRealizationAdmissionEvidence;
  input.providerRealizationAdmissionDispatchPolicy =
      record.providerRealizationAdmissionDispatchPolicy;
  input.providerRealizationAdmissionScheduleDecisionContract =
      record.providerRealizationAdmissionScheduleDecisionContract;
  input.providerRealizationAdmissionScheduleDecision =
      record.providerRealizationAdmissionScheduleDecision;
  input.providerRealizationAdmissionScheduleDecisionReason =
      record.providerRealizationAdmissionScheduleDecisionReason;
  input.providerPrimitiveChainContract =
      record.providerPrimitiveChainContract;
  input.providerPrimitiveChainKind = record.providerPrimitiveChainKind;
  input.providerPrimitiveContract = record.providerPrimitiveContract;
  input.providerPrimitiveKind = record.providerPrimitiveKind;
  input.providerWideningProductMultiplicandRoles =
      record.providerWideningProductMultiplicandRoles;
  input.providerWideningProductExtensionPolicy =
      record.providerWideningProductExtensionPolicy;
  input.providerPrimitiveSourceLoad = record.providerPrimitiveSourceLoad;
  input.providerPrimitiveSourceExtension =
      record.providerPrimitiveSourceExtension;
  input.providerPrimitiveSourceDType = record.providerPrimitiveSourceDType;
  input.providerPrimitiveSourceSignedness =
      record.providerPrimitiveSourceSignedness;
  input.providerPrimitiveSourceSEW = record.providerPrimitiveSourceSEW;
  input.providerPrimitiveSourceLMUL = record.providerPrimitiveSourceLMUL;
  input.providerPrimitiveProductDType = record.providerPrimitiveProductDType;
  input.providerPrimitiveProductSEW = record.providerPrimitiveProductSEW;
  input.providerPrimitiveProductLMUL = record.providerPrimitiveProductLMUL;
  input.providerPrimitiveAccumulatorDType =
      record.providerPrimitiveAccumulatorDType;
  input.providerPrimitiveAccumulatorSEW =
      record.providerPrimitiveAccumulatorSEW;
  input.providerPrimitiveAccumulatorLMUL =
      record.providerPrimitiveAccumulatorLMUL;
  input.providerPrimitiveResultDType = record.providerPrimitiveResultDType;
  input.providerPrimitiveResultSEW = record.providerPrimitiveResultSEW;
  input.providerPrimitiveResultLMUL = record.providerPrimitiveResultLMUL;
  input.providerPrimitiveWideningProductRelation =
      record.providerPrimitiveWideningProductRelation;
  input.providerPrimitiveProductReductionChainRelation =
      record.providerPrimitiveProductReductionChainRelation;
  input.providerPrimitiveWideningProductIntrinsic =
      record.providerPrimitiveWideningProductIntrinsic;
  input.providerPrimitiveReductionIntrinsic =
      record.providerPrimitiveReductionIntrinsic;
  input.providerPrimitiveScalarSeedSplatIntrinsic =
      record.providerPrimitiveScalarSeedSplatIntrinsic;
  input.providerPrimitiveAccumulatorLayout =
      record.providerPrimitiveAccumulatorLayout;
  input.providerPrimitiveResultLayout = record.providerPrimitiveResultLayout;
  input.providerPrimitiveReductionStoreVL =
      record.providerPrimitiveReductionStoreVL;
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
  if (llvm::Error error = requirePolicyString(
          context, "provider resource cost contract tie-back",
          outcome.providerResourceCostContract, selection.resourceCostContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider resource cost model tie-back",
          outcome.providerResourceCostModel, selection.resourceCostModel))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "provider resource cost loop-body steps tie-back",
          outcome.providerResourceCostLoopBodySteps,
          selection.resourceCostLoopBodySteps))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider resource cost blocker tie-back",
          outcome.providerResourceCostBlocker, selection.resourceCostBlocker))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider performance admission decision tie-back",
          outcome.providerPerformanceAdmissionDecision,
          selection.performanceAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider performance admission closure tie-back",
          outcome.providerPerformanceAdmissionClosure,
          selection.performanceAdmissionClosure))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider performance admission reopen requirement tie-back",
          outcome.providerPerformanceAdmissionReopenRequirement,
          selection.performanceAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider beyond-local repair admission contract tie-back",
          outcome.providerBeyondLocalRepairAdmissionContract,
          selection.beyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider beyond-local repair admission decision tie-back",
          outcome.providerBeyondLocalRepairAdmissionDecision,
          selection.beyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider beyond-local repair admission blocker tie-back",
          outcome.providerBeyondLocalRepairAdmissionBlocker,
          selection.beyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error = requirePolicyString(
          context,
          "provider beyond-local repair admission reopen requirement tie-back",
          outcome.providerBeyondLocalRepairAdmissionReopenRequirement,
          selection.beyondLocalRepairAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission contract tie-back",
          outcome.providerRealizationAdmissionContract,
          selection.realizationAdmissionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission decision tie-back",
          outcome.providerRealizationAdmissionDecision,
          selection.realizationAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission evidence tie-back",
          outcome.providerRealizationAdmissionEvidence,
          selection.realizationAdmissionEvidence))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "provider realization admission dispatch policy tie-back",
          outcome.providerRealizationAdmissionDispatchPolicy,
          selection.realizationAdmissionDispatchPolicy))
    return error;
  if (llvm::Error error = requirePolicyString(
          context,
          "provider realization admission schedule decision contract tie-back",
          outcome.providerRealizationAdmissionScheduleDecisionContract,
          selection.realizationAdmissionScheduleDecisionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context,
          "provider realization admission schedule decision tie-back",
          outcome.providerRealizationAdmissionScheduleDecision,
          selection.realizationAdmissionScheduleDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context,
          "provider realization admission schedule decision reason tie-back",
          outcome.providerRealizationAdmissionScheduleDecisionReason,
          selection.realizationAdmissionScheduleDecisionReason))
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
          getRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceIDForCandidate(
              selection.selectedCandidateID)))
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
  llvm::StringRef expectedBestSpeedupRange =
      getPackedI4Gate4MeasurementBestSpeedupRangeForCandidate(
          selection.selectedCandidateID);
  if (expectedBestSpeedupRange.empty())
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires candidate-sensitive Gate 4 same-target measurement range "
        "for selected packed-i4 candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requirePolicyString(
          context, "measurement best-speedup range",
          outcome.measurementBestSpeedupRange, expectedBestSpeedupRange))
    return error;
  const std::int64_t expectedSummaryRecordCount =
      getPackedI4Gate4MeasurementSummaryRecordCountForCandidate(
          selection.selectedCandidateID);
  if (expectedSummaryRecordCount <= 0)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires candidate-sensitive Gate 4 summary record count for "
        "selected packed-i4 candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requirePolicyInt(
          context, "measurement summary record count",
          outcome.measurementSummaryRecordCount, expectedSummaryRecordCount))
    return error;
  const std::int64_t expectedMeasurementRecordCount =
      getPackedI4Gate4MeasurementRecordCountForCandidate(
          selection.selectedCandidateID);
  if (expectedMeasurementRecordCount <= 0)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires candidate-sensitive Gate 4 measurement record count for "
        "selected packed-i4 candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requirePolicyInt(
          context, "measurement record count",
          outcome.measurementRecordCount, expectedMeasurementRecordCount))
    return error;
  const std::int64_t expectedCorrectnessRecordCount =
      getPackedI4Gate4CorrectnessRecordCountForCandidate(
          selection.selectedCandidateID);
  if (expectedCorrectnessRecordCount <= 0)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires candidate-sensitive Gate 4 correctness record count for "
        "selected packed-i4 candidate '" +
        selection.selectedCandidateID + "'");
  if (llvm::Error error = requirePolicyInt(
          context, "correctness record count", outcome.correctnessRecordCount,
          expectedCorrectnessRecordCount))
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
          getRVVLowPrecisionResourcePackedI4PerformanceBaselineForCandidate(
              selection.selectedCandidateID)))
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
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance admission decision",
          selection.performanceAdmissionDecision,
          kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance admission closure",
          selection.performanceAdmissionClosure,
          kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionClosure))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance admission reopen requirement",
          selection.performanceAdmissionReopenRequirement,
          kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission contract",
          selection.beyondLocalRepairAdmissionContract,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission decision",
          selection.beyondLocalRepairAdmissionDecision,
          kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission blocker",
          selection.beyondLocalRepairAdmissionBlocker,
          kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission reopen requirement",
          selection.beyondLocalRepairAdmissionReopenRequirement,
          kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionReopenRequirement))
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
         selection.dispatchPreference == "performance-preferred" ||
         selection.performanceAdmissionDecision ==
             kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionDecision ||
         selection.performanceAdmissionClosure ==
             kRVVLowPrecisionResourcePackedI4MeasuredWinPerformanceAdmissionClosure ||
         selection.beyondLocalRepairAdmissionDecision ==
             kRVVLowPrecisionResourcePackedI4MeasuredWinBeyondLocalRepairAdmissionDecision;
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
        " requires provider maturity outcome '" +
        kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome +
        "' for the accepted Gate 4 no-win measurement");
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
  if (llvm::StringRef(selection.performanceAdmissionDecision) !=
      kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider performance admission decision '" +
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision +
        "' for the accepted Gate 4 regression/no-win measurement");
  if (llvm::StringRef(selection.performanceAdmissionClosure) !=
      kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider performance admission closure '" +
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure +
        "' for the accepted Gate 4 no-safe-repair no-win measurement");
  if (llvm::StringRef(selection.performanceAdmissionReopenRequirement) !=
      kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider performance admission reopen requirement '" +
        kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement +
        "' before any future Gate 4 performance-preferred claim is reopened");
  if (llvm::StringRef(selection.beyondLocalRepairAdmissionContract) !=
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider beyond-local repair admission contract '" +
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract +
        "' for the accepted Gate 4 no-further-repair boundary");
  if (llvm::StringRef(selection.beyondLocalRepairAdmissionDecision) !=
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider beyond-local repair admission decision '" +
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision +
        "' for the accepted Gate 4 no-further-repair boundary");
  if (llvm::StringRef(selection.beyondLocalRepairAdmissionBlocker) !=
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider beyond-local repair admission blocker '" +
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker +
        "' for the accepted Gate 4 no-further-repair boundary");
  if (llvm::StringRef(selection.beyondLocalRepairAdmissionReopenRequirement) !=
      kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " requires provider beyond-local repair admission reopen requirement '" +
        kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement +
        "' before any future Gate 4 beyond-local repair claim is reopened");
  return llvm::Error::success();
}

llvm::Error verifyNoWinSelectedDispatchPreferenceDenial(
    const RVVLowPrecisionPerformancePolicyDecision &decision,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  if (llvm::Error error =
          requirePolicyString(context, "no-win dispatch preference",
                              decision.dispatchPreference,
                              kRVVLowPrecisionResourcePackedI4DispatchPreference))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "no-win performance preference denial reason",
          decision.performancePreferenceDenialReason,
          kPackedI4PerformancePreferenceDenialReason))
    return error;
  if (llvm::Error error =
          rejectNoWinDispatchPerformancePreferenceMarker(
              context, "selected dispatch case policy",
              dispatchBoundary.selectedCasePolicy))
    return error;
  if (llvm::Error error =
          rejectNoWinDispatchPerformancePreferenceMarker(
              context, "selected dispatch case mirror",
              dispatchBoundary.selectedDispatchCaseMirror))
    return error;
  if (llvm::Error error =
          rejectNoWinDispatchPerformancePreferenceMarker(
              context, "selected dispatch fallback policy",
              dispatchBoundary.fallbackPolicy))
    return error;
  return rejectNoWinDispatchPerformancePreferenceMarker(
      context, "selected dispatch fallback mirror",
      dispatchBoundary.selectedDispatchFallbackMirror);
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
    return verifyRVVLowPrecisionSelectedDispatchMirrorTieBack(dispatchBoundary,
                                                            context);
  }
  if (!decision.correctnessFallbackPathSelected ||
      decision.dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " no-win/regression selected-dispatch policy requires the "
        "correctness-fallback dispatch path");
  if (llvm::Error error =
          verifyNoWinSelectedDispatchPreferenceDenial(decision,
                                                     dispatchBoundary, context))
    return error;
  return verifyRVVLowPrecisionSelectedDispatchMirrorTieBack(dispatchBoundary,
                                                          context);
}

llvm::Expected<RVVLowPrecisionProductionPressureProfile>
materializeRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    const RVVLowPrecisionPerformancePolicyDecision &decision,
    llvm::StringRef context) {
  if (llvm::Error error =
          verifyPackedI4SameTargetMeasurementPolicyInput(selection, input,
                                                         context))
    return std::move(error);
  if (llvm::Error error = rejectPressureProfileMarkerOnlyFacts(
          selection, input, dispatchBoundary, context))
    return std::move(error);
  if (llvm::Error error =
          verifyRVVLowPrecisionSelectedDispatchBoundary(decision,
                                                        dispatchBoundary,
                                                        context))
    return std::move(error);

  RVVLowPrecisionProductionPressureProfile profile;
  profile.contract = kProductionPressureProfileContract.str();
  profile.authority = kProductionPressureProfileAuthority.str();
  profile.pressurePath = selection.primitiveKind;
  profile.selectedCandidateID = selection.selectedCandidateID;
  profile.measurementProviderCandidateID =
      input.providerResourceSelectedCandidate;
  profile.resourcePlanningContract = input.providerResourcePlanningContract;
  profile.resourceOperandForm = input.providerResourceOperandForm;
  profile.resourceSourceSignedness = input.providerResourceSourceSignedness;
  profile.resourceStorageElementWidth =
      input.providerResourceStorageElementWidth;
  profile.resourceEffectiveElementWidth =
      input.providerResourceEffectiveElementWidth;
  profile.resourcePackingLayout = input.providerResourcePackingLayout;
  profile.resourceUnpackIntent = input.providerResourceUnpackIntent;
  profile.vsetvlRegionCount = input.providerResourceVSetVLRegionCount;
  profile.runtimeAVLSource = input.providerRuntimeAVLSource;
  profile.runtimeABIOrder = input.providerRuntimeABIOrder;
  profile.routeFamilyPlan = input.providerResourceRouteFamilyPlan;
  profile.providerSupportedMirror = input.providerSupportedMirror;
  profile.sourceRecordContract = input.sourceRecordContract;
  profile.sourceSelectedVariant = input.sourceSelectedVariant;
  profile.sourceSelectedInput = input.sourceSelectedInput;
  profile.sourceGeneratedFunction = input.sourceGeneratedFunction;
  profile.generatedArtifactIdentityContract =
      input.generatedArtifactIdentityContract;
  profile.generatedArtifactObjectPath = input.generatedArtifactObjectPath;
  profile.generatedArtifactObjectSHA256 = input.generatedArtifactObjectSHA256;
  profile.generatedArtifactHeaderPath = input.generatedArtifactHeaderPath;
  profile.generatedArtifactHeaderSHA256 = input.generatedArtifactHeaderSHA256;
  profile.measurementTarget = input.measurementTarget;
  profile.measurementTargetProvenance = input.measurementTargetProvenance;
  profile.measurementRuntimeCountSet = input.measurementRuntimeCountSet;
  profile.measurementRuntimeCountProvenance =
      input.measurementRuntimeCountProvenance;
  profile.pressureProfileLabel = input.pressureProfileLabel;
  profile.pressureProfileLabelProvenance =
      input.pressureProfileLabelProvenance;

  profile.primitiveChainContract = input.providerPrimitiveChainContract;
  profile.primitiveChainKind = input.providerPrimitiveChainKind;
  profile.primitiveContract = input.providerPrimitiveContract;
  profile.primitiveKind = input.providerPrimitiveKind;
  profile.wideningProductMultiplicandRoles =
      input.providerWideningProductMultiplicandRoles;
  profile.wideningProductExtensionPolicy =
      input.providerWideningProductExtensionPolicy;
  profile.primitiveSourceLoad = input.providerPrimitiveSourceLoad;
  profile.primitiveSourceExtension = input.providerPrimitiveSourceExtension;
  profile.primitiveSourceDType = input.providerPrimitiveSourceDType;
  profile.primitiveSourceSignedness = input.providerPrimitiveSourceSignedness;
  profile.primitiveSourceSEW = input.providerPrimitiveSourceSEW;
  profile.primitiveSourceLMUL = input.providerPrimitiveSourceLMUL;
  profile.primitiveProductDType = input.providerPrimitiveProductDType;
  profile.primitiveProductSEW = input.providerPrimitiveProductSEW;
  profile.primitiveProductLMUL = input.providerPrimitiveProductLMUL;
  profile.primitiveAccumulatorDType =
      input.providerPrimitiveAccumulatorDType;
  profile.primitiveAccumulatorSEW =
      input.providerPrimitiveAccumulatorSEW;
  profile.primitiveAccumulatorLMUL =
      input.providerPrimitiveAccumulatorLMUL;
  profile.primitiveResultDType = input.providerPrimitiveResultDType;
  profile.primitiveResultSEW = input.providerPrimitiveResultSEW;
  profile.primitiveResultLMUL = input.providerPrimitiveResultLMUL;
  profile.primitiveWideningProductRelation =
      input.providerPrimitiveWideningProductRelation;
  profile.primitiveProductReductionChainRelation =
      input.providerPrimitiveProductReductionChainRelation;
  profile.primitiveWideningProductIntrinsic =
      input.providerPrimitiveWideningProductIntrinsic;
  profile.primitiveReductionIntrinsic =
      input.providerPrimitiveReductionIntrinsic;
  profile.primitiveScalarSeedSplatIntrinsic =
      input.providerPrimitiveScalarSeedSplatIntrinsic;
  profile.primitiveAccumulatorLayout =
      input.providerPrimitiveAccumulatorLayout;
  profile.primitiveResultLayout = input.providerPrimitiveResultLayout;
  profile.primitiveReductionStoreVL =
      input.providerPrimitiveReductionStoreVL;

  profile.scheduleDecisionContract =
      input.providerScheduleDecisionContract;
  profile.scheduleDecision = input.providerScheduleDecision;
  profile.scheduleDecisionReason = input.providerScheduleDecisionReason;
  profile.resourceCostContract = input.providerResourceCostContract;
  profile.resourceCostModel = input.providerResourceCostModel;
  profile.resourceCostLoopBodySteps = input.providerResourceCostLoopBodySteps;
  profile.resourceCostBlocker = input.providerResourceCostBlocker;
  profile.performanceAdmissionDecision =
      input.providerPerformanceAdmissionDecision;
  profile.performanceAdmissionClosure =
      input.providerPerformanceAdmissionClosure;
  profile.performanceAdmissionReopenRequirement =
      input.providerPerformanceAdmissionReopenRequirement;
  profile.beyondLocalRepairAdmissionContract =
      input.providerBeyondLocalRepairAdmissionContract;
  profile.beyondLocalRepairAdmissionDecision =
      input.providerBeyondLocalRepairAdmissionDecision;
  profile.beyondLocalRepairAdmissionBlocker =
      input.providerBeyondLocalRepairAdmissionBlocker;
  profile.beyondLocalRepairAdmissionReopenRequirement =
      input.providerBeyondLocalRepairAdmissionReopenRequirement;
  profile.realizationAdmissionContract =
      input.providerRealizationAdmissionContract;
  profile.realizationAdmissionDecision =
      input.providerRealizationAdmissionDecision;
  profile.realizationAdmissionEvidence =
      input.providerRealizationAdmissionEvidence;
  profile.realizationAdmissionDispatchPolicy =
      input.providerRealizationAdmissionDispatchPolicy;
  profile.realizationAdmissionScheduleDecisionContract =
      input.providerRealizationAdmissionScheduleDecisionContract;
  profile.realizationAdmissionScheduleDecision =
      input.providerRealizationAdmissionScheduleDecision;
  profile.realizationAdmissionScheduleDecisionReason =
      input.providerRealizationAdmissionScheduleDecisionReason;
  profile.targetProfile = input.targetProfile;
  profile.targetCapabilityProviderMirror =
      input.targetCapabilityProviderMirror;
  profile.targetCapabilityLegalityMirror =
      input.targetCapabilityLegalityMirror;
  profile.measurementEvidenceID = input.measurementEvidenceID;
  profile.measurementClassification = input.measurementClassification;
  profile.measurementOutcomeFamily = input.measurementOutcomeFamily;
  profile.measurementBestSpeedupRange = input.measurementBestSpeedupRange;
  profile.measurementSummaryRecordCount =
      input.measurementSummaryRecordCount;
  profile.measurementRecordCount = input.measurementRecordCount;
  profile.correctnessRecordCount = input.correctnessRecordCount;

  profile.selectedDispatchCaseMirror =
      dispatchBoundary.selectedDispatchCaseMirror;
  profile.selectedDispatchFallbackMirror =
      dispatchBoundary.selectedDispatchFallbackMirror;
  profile.selectedCaseVariant = dispatchBoundary.selectedCaseVariant;
  profile.fallbackVariant = dispatchBoundary.fallbackVariant;

  profile.dispatchPolicyPath = decision.dispatchPolicyPath;
  profile.dispatchPreference = decision.dispatchPreference;
  profile.routeSupportAllowed = decision.routeSupportAllowed;
  profile.correctnessExecutionAllowed = decision.correctnessExecutionAllowed;
  profile.performanceSelectionAllowed = decision.performanceSelectionAllowed;
  profile.performanceWinClaimAllowed = decision.performanceWinClaimAllowed;
  profile.correctnessFallbackPathSelected =
      decision.correctnessFallbackPathSelected;
  profile.performancePreferredPathSelected =
      decision.performancePreferredPathSelected;
  return profile;
}

llvm::Error rejectProductionPressureProfileMarkerOnlyFacts(
    const RVVLowPrecisionProductionPressureProfile &profile,
    llvm::StringRef context) {
  struct Field {
    llvm::StringRef label;
    llvm::StringRef value;
  };
  Field fields[] = {
      {"pressure profile authority", profile.authority},
      {"pressure path", profile.pressurePath},
      {"selected candidate", profile.selectedCandidateID},
      {"measurement provider candidate",
       profile.measurementProviderCandidateID},
      {"resource planning contract", profile.resourcePlanningContract},
      {"resource operand form", profile.resourceOperandForm},
      {"resource packing layout", profile.resourcePackingLayout},
      {"resource unpack intent", profile.resourceUnpackIntent},
      {"runtime AVL source", profile.runtimeAVLSource},
      {"runtime ABI order", profile.runtimeABIOrder},
      {"route-family plan", profile.routeFamilyPlan},
      {"provider-supported mirror", profile.providerSupportedMirror},
      {"source record contract", profile.sourceRecordContract},
      {"source selected variant", profile.sourceSelectedVariant},
      {"source selected input", profile.sourceSelectedInput},
      {"source generated function", profile.sourceGeneratedFunction},
      {"generated artifact object path",
       profile.generatedArtifactObjectPath},
      {"generated artifact object sha256",
       profile.generatedArtifactObjectSHA256},
      {"generated artifact header path",
       profile.generatedArtifactHeaderPath},
      {"generated artifact header sha256",
       profile.generatedArtifactHeaderSHA256},
      {"measurement target", profile.measurementTarget},
      {"measurement target provenance", profile.measurementTargetProvenance},
      {"measurement runtime count set",
       profile.measurementRuntimeCountSet},
      {"measurement runtime count provenance",
       profile.measurementRuntimeCountProvenance},
      {"pressure profile label", profile.pressureProfileLabel},
      {"pressure profile label provenance",
       profile.pressureProfileLabelProvenance},
      {"primitive chain contract", profile.primitiveChainContract},
      {"primitive kind", profile.primitiveKind},
      {"widening product multiplicand roles",
       profile.wideningProductMultiplicandRoles},
      {"widening product extension policy",
       profile.wideningProductExtensionPolicy},
      {"primitive source load", profile.primitiveSourceLoad},
      {"primitive source extension", profile.primitiveSourceExtension},
      {"primitive source dtype", profile.primitiveSourceDType},
      {"primitive source signedness", profile.primitiveSourceSignedness},
      {"primitive product dtype", profile.primitiveProductDType},
      {"primitive accumulator dtype", profile.primitiveAccumulatorDType},
      {"primitive result dtype", profile.primitiveResultDType},
      {"primitive widening product relation",
       profile.primitiveWideningProductRelation},
      {"primitive product-reduction relation",
       profile.primitiveProductReductionChainRelation},
      {"primitive widening product intrinsic",
       profile.primitiveWideningProductIntrinsic},
      {"primitive reduction intrinsic", profile.primitiveReductionIntrinsic},
      {"primitive scalar seed splat intrinsic",
       profile.primitiveScalarSeedSplatIntrinsic},
      {"primitive accumulator layout", profile.primitiveAccumulatorLayout},
      {"primitive result layout", profile.primitiveResultLayout},
      {"primitive reduction store VL", profile.primitiveReductionStoreVL},
      {"schedule decision contract", profile.scheduleDecisionContract},
      {"schedule decision", profile.scheduleDecision},
      {"schedule decision reason", profile.scheduleDecisionReason},
      {"resource cost contract", profile.resourceCostContract},
      {"resource cost model", profile.resourceCostModel},
      {"resource cost blocker", profile.resourceCostBlocker},
      {"performance admission decision",
       profile.performanceAdmissionDecision},
      {"performance admission closure", profile.performanceAdmissionClosure},
      {"performance admission reopen requirement",
       profile.performanceAdmissionReopenRequirement},
      {"beyond-local repair admission contract",
       profile.beyondLocalRepairAdmissionContract},
      {"beyond-local repair admission decision",
       profile.beyondLocalRepairAdmissionDecision},
      {"beyond-local repair admission blocker",
       profile.beyondLocalRepairAdmissionBlocker},
      {"beyond-local repair admission reopen requirement",
       profile.beyondLocalRepairAdmissionReopenRequirement},
      {"realization admission contract",
       profile.realizationAdmissionContract},
      {"realization admission decision",
       profile.realizationAdmissionDecision},
      {"realization admission evidence",
       profile.realizationAdmissionEvidence},
      {"realization admission dispatch policy",
       profile.realizationAdmissionDispatchPolicy},
      {"realization admission schedule decision contract",
       profile.realizationAdmissionScheduleDecisionContract},
      {"realization admission schedule decision",
       profile.realizationAdmissionScheduleDecision},
      {"realization admission schedule decision reason",
       profile.realizationAdmissionScheduleDecisionReason},
      {"target profile", profile.targetProfile},
      {"target capability provider mirror",
       profile.targetCapabilityProviderMirror},
      {"target capability legality mirror",
       profile.targetCapabilityLegalityMirror},
      {"measurement evidence id", profile.measurementEvidenceID},
      {"selected dispatch case mirror",
       profile.selectedDispatchCaseMirror},
      {"selected dispatch fallback mirror",
       profile.selectedDispatchFallbackMirror},
      {"selected dispatch case variant", profile.selectedCaseVariant},
      {"fallback variant", profile.fallbackVariant},
      {"dispatch policy path", profile.dispatchPolicyPath},
      {"dispatch preference", profile.dispatchPreference},
  };
  for (const Field &field : fields) {
    if (llvm::Error error =
            rejectLabelOnlyPressureMarker(context, field.label, field.value))
      return error;
    if (llvm::Error error =
            rejectMetadataOnlyPressureMarker(context, field.label, field.value))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error verifyProductionPressureProfileDispatchAdmission(
    const RVVLowPrecisionProductionPressureProfile &profile,
    llvm::StringRef context) {
  if (llvm::Error error =
          requirePolicyBool(context, "route support allowed",
                            profile.routeSupportAllowed, true))
    return error;
  if (llvm::Error error =
          requirePolicyBool(context, "correctness execution allowed",
                            profile.correctnessExecutionAllowed, true))
    return error;
  if (profile.dispatchPolicyPath == kPackedI4CorrectnessFallbackPolicyPath) {
    if (llvm::Error error =
            requirePolicyBool(context, "correctness fallback path selected",
                              profile.correctnessFallbackPathSelected, true))
      return error;
    if (llvm::Error error =
            requirePolicyBool(context, "performance preferred path selected",
                              profile.performancePreferredPathSelected, false))
      return error;
    return llvm::Error::success();
  }
  if (profile.dispatchPolicyPath == kPackedI4PerformancePreferredPolicyPath) {
    if (llvm::Error error =
            requirePolicyBool(context, "performance preferred path selected",
                              profile.performancePreferredPathSelected, true))
      return error;
    if (llvm::Error error =
            requirePolicyBool(context, "performance selection allowed",
                              profile.performanceSelectionAllowed, true))
      return error;
    if (llvm::Error error =
            requirePolicyBool(context, "performance win claim allowed",
                              profile.performanceWinClaimAllowed, true))
      return error;
    return llvm::Error::success();
  }
  return makeRVVLowPrecisionPerformancePolicyError(
      llvm::Twine(context) +
      " selected-body realization admission rejects unsupported dispatch "
      "policy path '" +
      profile.dispatchPolicyPath +
      "'; expected correctness-fallback or performance-preferred");
}

llvm::Error verifyProductionPressureProfileAgainstCandidate(
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVLowPrecisionProductionPressureProfile &profile,
    llvm::StringRef context) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(candidate.candidateID))
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-body realization admission requires a source-backed "
        "packed-i4 low-precision production pressure profile for the current "
        "Gate 1 boundary");
  if (llvm::Error error =
          requirePolicyString(context, "packed-i4 schedule decision contract",
                              candidate.scheduleDecisionContract,
                              kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "packed-i4 schedule decision",
                              candidate.scheduleDecision,
                              kRVVLowPrecisionResourcePackedI4ScheduleDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 schedule decision reason",
          candidate.scheduleDecisionReason,
          kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 resource cost contract",
          candidate.resourceCostContract,
          kRVVLowPrecisionResourcePackedI4CostContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 resource cost model",
          candidate.resourceCostModel, kRVVLowPrecisionResourcePackedI4CostModel))
    return error;
  if (llvm::Error error = requirePolicyInt(
          context, "packed-i4 resource cost loop-body steps",
          candidate.resourceCostLoopBodySteps,
          kRVVLowPrecisionResourcePackedI4CostLoopBodySteps))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 resource cost blocker",
          candidate.resourceCostBlocker,
          kRVVLowPrecisionResourcePackedI4CostBlocker))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance admission decision",
          candidate.performanceAdmissionDecision,
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance admission closure",
          candidate.performanceAdmissionClosure,
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionClosure))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 performance admission reopen requirement",
          candidate.performanceAdmissionReopenRequirement,
          kRVVLowPrecisionResourcePackedI4PerformanceAdmissionReopenRequirement))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission contract",
          candidate.beyondLocalRepairAdmissionContract,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission decision",
          candidate.beyondLocalRepairAdmissionDecision,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission blocker",
          candidate.beyondLocalRepairAdmissionBlocker,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "packed-i4 beyond-local repair admission reopen requirement",
          candidate.beyondLocalRepairAdmissionReopenRequirement,
          kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "pressure profile contract",
                              profile.contract,
                              kProductionPressureProfileContract))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "pressure profile authority",
                              profile.authority,
                              kProductionPressureProfileAuthority))
    return error;
  if (llvm::Error error =
          rejectProductionPressureProfileMarkerOnlyFacts(profile, context))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "selected candidate", profile.selectedCandidateID,
          candidate.candidateID))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement provider candidate",
                              profile.measurementProviderCandidateID,
                              candidate.candidateID))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "resource planning contract",
                              profile.resourcePlanningContract,
                              candidate.planningContract))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "resource operand form",
                              profile.resourceOperandForm,
                              candidate.operandForm))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "resource source signedness",
                              profile.resourceSourceSignedness,
                              candidate.sourceSignedness))
    return error;
  if (llvm::Error error =
          requirePolicyInt(context, "resource storage element width",
                           profile.resourceStorageElementWidth,
                           candidate.storageElementWidth))
    return error;
  if (llvm::Error error =
          requirePolicyInt(context, "resource effective element width",
                           profile.resourceEffectiveElementWidth,
                           candidate.effectiveElementWidth))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "resource packing layout",
                              profile.resourcePackingLayout,
                              candidate.packingLayout))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "resource unpack intent",
                              profile.resourceUnpackIntent,
                              candidate.unpackIntent))
    return error;
  if (llvm::Error error =
          requirePolicyInt(context, "resource vsetvl region count",
                           profile.vsetvlRegionCount,
                           candidate.vsetvlRegionCount))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "runtime AVL source",
                              profile.runtimeAVLSource,
                              candidate.runtimeAVLSource))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "runtime ABI order",
                              profile.runtimeABIOrder,
                              candidate.runtimeABIOrder))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "route-family plan",
                                      profile.routeFamilyPlan))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "provider-supported mirror",
                                      profile.providerSupportedMirror))
    return error;

  if (llvm::Error error =
          requirePolicyString(context, "source record contract",
                              profile.sourceRecordContract,
                              kSourceBackedMeasurementRecordContract))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "source selected variant",
                              profile.sourceSelectedVariant,
                              getPackedI4SourceSelectedVariantForCandidate(
                                  candidate.candidateID)))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "source selected input",
                              profile.sourceSelectedInput,
                              getPackedI4SourceSelectedInputForCandidate(
                                  candidate.candidateID)))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "source generated function",
                              profile.sourceGeneratedFunction,
                              getPackedI4SourceGeneratedFunctionForCandidate(
                                  candidate.candidateID)))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "generated artifact identity contract",
                              profile.generatedArtifactIdentityContract,
                              kGeneratedArtifactIdentityContract))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "generated artifact object path",
                                      profile.generatedArtifactObjectPath))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "generated artifact object sha256",
                                      profile.generatedArtifactObjectSHA256))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "generated artifact header path",
                                      profile.generatedArtifactHeaderPath))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "generated artifact header sha256",
                                      profile.generatedArtifactHeaderSHA256))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement target",
                              profile.measurementTarget, profile.targetProfile))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement target provenance",
                              profile.measurementTargetProvenance,
                              kMeasurementTargetProvenance))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "measurement runtime count set",
                                      profile.measurementRuntimeCountSet))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "measurement runtime count provenance",
                              profile.measurementRuntimeCountProvenance,
                              kMeasurementRuntimeCountProvenance))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "pressure profile label",
                              profile.pressureProfileLabel,
                              kProductionPressureProfileLabel))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "pressure profile label provenance",
                              profile.pressureProfileLabelProvenance,
                              kProductionPressureProfileLabelProvenance))
    return error;

  if (llvm::Error error =
          requirePolicyString(context, "primitive chain contract",
                              profile.primitiveChainContract,
                              candidate.primitiveChainContractID))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive chain kind",
                              profile.primitiveChainKind,
                              candidate.primitiveChainKind))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive contract",
                              profile.primitiveContract,
                              candidate.primitiveContractID))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive kind", profile.primitiveKind,
                              candidate.primitiveKind))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "widening product multiplicand roles",
          profile.wideningProductMultiplicandRoles,
          candidate.wideningProductMultiplicandRoleSummary))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "widening product extension policy",
          profile.wideningProductExtensionPolicy,
          candidate.wideningProductExtensionPolicy))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive source load",
                              profile.primitiveSourceLoad,
                              candidate.primitiveSourceLoadKind))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive source extension",
                              profile.primitiveSourceExtension,
                              candidate.primitiveSourceExtensionKind))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive source dtype",
                              profile.primitiveSourceDType,
                              candidate.sourceElementTypeName))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive source signedness",
                              profile.primitiveSourceSignedness,
                              candidate.sourceSignedness))
    return error;
  if (llvm::Error error =
          requirePolicyInt(context, "primitive source SEW",
                           profile.primitiveSourceSEW, candidate.sourceSEW))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive source LMUL",
                              profile.primitiveSourceLMUL,
                              candidate.sourceLMUL))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive product dtype",
                              profile.primitiveProductDType,
                              candidate.productElementTypeName))
    return error;
  if (llvm::Error error =
          requirePolicyInt(context, "primitive product SEW",
                           profile.primitiveProductSEW, candidate.productSEW))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive product LMUL",
                              profile.primitiveProductLMUL,
                              candidate.productLMUL))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive accumulator dtype",
                              profile.primitiveAccumulatorDType,
                              candidate.accumulatorElementTypeName))
    return error;
  if (llvm::Error error =
          requirePolicyInt(context, "primitive accumulator SEW",
                           profile.primitiveAccumulatorSEW,
                           candidate.accumulatorSEW))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive accumulator LMUL",
                              profile.primitiveAccumulatorLMUL,
                              candidate.accumulatorLMUL))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive result dtype",
                              profile.primitiveResultDType,
                              candidate.resultElementTypeName))
    return error;
  if (llvm::Error error =
          requirePolicyInt(context, "primitive result SEW",
                           profile.primitiveResultSEW, candidate.resultSEW))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive result LMUL",
                              profile.primitiveResultLMUL,
                              candidate.resultLMUL))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive widening product relation",
                              profile.primitiveWideningProductRelation,
                              candidate.primitiveWideningProductRelation))
    return error;
  if (llvm::Error error =
          requirePolicyString(context,
                              "primitive product-reduction chain relation",
                              profile.primitiveProductReductionChainRelation,
                              candidate
                                  .primitiveProductReductionChainRelation))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive widening product intrinsic",
                              profile.primitiveWideningProductIntrinsic,
                              candidate.primitiveWideningProductIntrinsic))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive reduction intrinsic",
                              profile.primitiveReductionIntrinsic,
                              candidate.primitiveReductionIntrinsic))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive scalar seed splat intrinsic",
                              profile.primitiveScalarSeedSplatIntrinsic,
                              candidate.primitiveScalarSeedSplatIntrinsic))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive accumulator layout",
                              profile.primitiveAccumulatorLayout,
                              candidate.primitiveAccumulatorLayout))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive result layout",
                              profile.primitiveResultLayout,
                              candidate.primitiveResultLayout))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "primitive reduction store VL",
                              profile.primitiveReductionStoreVL,
                              candidate.primitiveReductionStoreVL))
    return error;

  if (llvm::Error error =
          requirePolicyString(context, "schedule decision contract",
                              profile.scheduleDecisionContract,
                              candidate.scheduleDecisionContract))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "schedule decision",
                              profile.scheduleDecision,
                              candidate.scheduleDecision))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "schedule decision reason",
                              profile.scheduleDecisionReason,
                              candidate.scheduleDecisionReason))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission contract",
          profile.realizationAdmissionContract,
          kSelectedBodyRealizationAdmissionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission decision",
          profile.realizationAdmissionDecision,
          stringifyRVVLowPrecisionRealizationAdmissionDecision(
              RVVLowPrecisionRealizationAdmissionDecision::Realize)))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "realization admission evidence",
                              profile.realizationAdmissionEvidence,
                              profile.measurementEvidenceID))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission dispatch policy",
          profile.realizationAdmissionDispatchPolicy,
          profile.dispatchPolicyPath))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission schedule decision contract",
          profile.realizationAdmissionScheduleDecisionContract,
          profile.scheduleDecisionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission schedule decision",
          profile.realizationAdmissionScheduleDecision,
          profile.scheduleDecision))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission schedule decision reason",
          profile.realizationAdmissionScheduleDecisionReason,
          profile.scheduleDecisionReason))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "target profile",
                                      profile.targetProfile))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context,
                                      "target capability provider mirror",
                                      profile.targetCapabilityProviderMirror))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context,
                                      "target capability legality mirror",
                                      profile.targetCapabilityLegalityMirror))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "measurement evidence id",
                                      profile.measurementEvidenceID))
    return error;
  if (llvm::Error error =
          requirePositivePolicyInt(context, "measurement summary record count",
                                   profile.measurementSummaryRecordCount))
    return error;
  if (llvm::Error error =
          requirePositivePolicyInt(context, "measurement record count",
                                   profile.measurementRecordCount))
    return error;
  if (llvm::Error error =
          requirePositivePolicyInt(context, "correctness record count",
                                   profile.correctnessRecordCount))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "selected dispatch case mirror",
                                      profile.selectedDispatchCaseMirror))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context,
                                      "selected dispatch fallback mirror",
                                      profile.selectedDispatchFallbackMirror))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "selected dispatch case variant",
                              profile.selectedCaseVariant,
                              profile.sourceSelectedVariant))
    return error;
  if (llvm::Error error =
          requireNonEmptyPolicyString(context, "fallback variant",
                                      profile.fallbackVariant))
    return error;
  return verifyProductionPressureProfileDispatchAdmission(profile, context);
}

llvm::Error verifyProductionPressureProfileAgainstSelection(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionProductionPressureProfile &profile,
    llvm::StringRef context) {
  if (!selection.hasSelection)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-body realization admission requires provider-selected "
        "low-precision resource facts");
  RVVLowPrecisionContractionResourceCandidate candidate;
  candidate.candidateSetID = selection.candidateSetID;
  candidate.candidateID = selection.selectedCandidateID;
  candidate.selectionReason = selection.selectionReason;
  candidate.planningContract = selection.planningContract;
  candidate.legalityScope = selection.legalityScope;
  candidate.sourceElementTypeName = selection.sourceElementTypeName;
  candidate.sourceSEW = selection.sourceSEW;
  candidate.sourceLMUL = selection.sourceLMUL;
  candidate.operandForm = selection.operandForm;
  candidate.sourceSignedness = selection.sourceSignedness;
  candidate.storageElementWidth = selection.storageElementWidth;
  candidate.effectiveElementWidth = selection.effectiveElementWidth;
  candidate.packingLayout = selection.packingLayout;
  candidate.unpackIntent = selection.unpackIntent;
  candidate.productElementTypeName = selection.productElementTypeName;
  candidate.productSEW = selection.productSEW;
  candidate.productLMUL = selection.productLMUL;
  candidate.productEMUL = selection.productEMUL;
  candidate.accumulatorElementTypeName =
      selection.accumulatorElementTypeName;
  candidate.accumulatorSEW = selection.accumulatorSEW;
  candidate.accumulatorLMUL = selection.accumulatorLMUL;
  candidate.accumulatorEMUL = selection.accumulatorEMUL;
  candidate.resultElementTypeName = selection.resultElementTypeName;
  candidate.resultSEW = selection.resultSEW;
  candidate.resultLMUL = selection.resultLMUL;
  candidate.memoryForm = selection.memoryForm;
  candidate.tailPolicy = selection.tailPolicy;
  candidate.maskPolicy = selection.maskPolicy;
  candidate.unrollFactor = selection.unrollFactor;
  candidate.accumulatorCount = selection.accumulatorCount;
  candidate.reductionLayout = selection.reductionLayout;
  candidate.vsetvlRegionCount = selection.vsetvlRegionCount;
  candidate.peakLiveVectorGroups = selection.peakLiveVectorGroups;
  candidate.vectorRegisterBudget = selection.vectorRegisterBudget;
  candidate.runtimeAVLSource = selection.runtimeAVLSource;
  candidate.producerScope = selection.producerScope;
  candidate.consumerScope = selection.consumerScope;
  candidate.runtimeABIOrder = selection.runtimeABIOrder;
  candidate.primitiveContractID = selection.primitiveContractID;
  candidate.primitiveKind = selection.primitiveKind;
  candidate.primitiveChainContractID = selection.primitiveChainContractID;
  candidate.primitiveChainKind = selection.primitiveChainKind;
  candidate.wideningProductMultiplicandRoleSummary =
      selection.wideningProductMultiplicandRoleSummary;
  candidate.wideningProductExtensionPolicy =
      selection.wideningProductExtensionPolicy;
  candidate.primitiveSourceLoadKind = selection.primitiveSourceLoadKind;
  candidate.primitiveSourceExtensionKind =
      selection.primitiveSourceExtensionKind;
  candidate.primitiveWideningProductRelation =
      selection.primitiveWideningProductRelation;
  candidate.primitiveProductReductionChainRelation =
      selection.primitiveProductReductionChainRelation;
  candidate.primitiveWideningProductIntrinsic =
      selection.primitiveWideningProductIntrinsic;
  candidate.primitiveReductionIntrinsic =
      selection.primitiveReductionIntrinsic;
  candidate.primitiveScalarSeedSplatIntrinsic =
      selection.primitiveScalarSeedSplatIntrinsic;
  candidate.primitiveAccumulatorLayout = selection.primitiveAccumulatorLayout;
  candidate.primitiveResultLayout = selection.primitiveResultLayout;
  candidate.primitiveReductionStoreVL = selection.primitiveReductionStoreVL;
  candidate.scheduleDecisionContract = selection.scheduleDecisionContract;
  candidate.scheduleDecision = selection.scheduleDecision;
  candidate.scheduleDecisionReason = selection.scheduleDecisionReason;
  candidate.resourceCostContract = selection.resourceCostContract;
  candidate.resourceCostModel = selection.resourceCostModel;
  candidate.resourceCostLoopBodySteps = selection.resourceCostLoopBodySteps;
  candidate.resourceCostBlocker = selection.resourceCostBlocker;
  candidate.performanceAdmissionDecision =
      selection.performanceAdmissionDecision;
  candidate.performanceAdmissionClosure = selection.performanceAdmissionClosure;
  candidate.performanceAdmissionReopenRequirement =
      selection.performanceAdmissionReopenRequirement;
  candidate.beyondLocalRepairAdmissionContract =
      selection.beyondLocalRepairAdmissionContract;
  candidate.beyondLocalRepairAdmissionDecision =
      selection.beyondLocalRepairAdmissionDecision;
  candidate.beyondLocalRepairAdmissionBlocker =
      selection.beyondLocalRepairAdmissionBlocker;
  candidate.beyondLocalRepairAdmissionReopenRequirement =
      selection.beyondLocalRepairAdmissionReopenRequirement;
  candidate.isLegal = selection.isLegal;
  candidate.rejectionReason = selection.rejectionReason;
  if (llvm::Error error =
          verifyProductionPressureProfileAgainstCandidate(candidate, profile,
                                                         context))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "route-family plan",
                              profile.routeFamilyPlan,
                              selection.routeFamilyPlanID))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "provider-supported mirror",
                              profile.providerSupportedMirror,
                              selection.providerSupportedMirror))
    return error;
  if (llvm::Error error =
          requirePolicyString(context, "target capability provider mirror",
                              profile.targetCapabilityProviderMirror,
                              selection.targetCapabilityProviderMirror))
    return error;
  return requirePolicyString(context, "target capability legality mirror",
                             profile.targetCapabilityLegalityMirror,
                             selection.targetCapabilityLegalityMirror);
}

RVVLowPrecisionSelectedBodyRealizationAdmission
makeSelectedBodyRealizationAdmission(
    const RVVLowPrecisionProductionPressureProfile &profile) {
  RVVLowPrecisionSelectedBodyRealizationAdmission admission;
  admission.admissionContract =
      kSelectedBodyRealizationAdmissionContract.str();
  admission.admissionOwner = kSelectedBodyRealizationAdmissionOwner.str();
  admission.decision =
      RVVLowPrecisionRealizationAdmissionDecision::Realize;
  admission.selectedCandidateID = profile.selectedCandidateID;
  admission.pressureProfileContract = profile.contract;
  admission.measurementEvidenceID = profile.measurementEvidenceID;
  admission.dispatchPolicyPath = profile.dispatchPolicyPath;
  admission.scheduleDecisionContract = profile.scheduleDecisionContract;
  admission.scheduleDecision = profile.scheduleDecision;
  admission.scheduleDecisionReason = profile.scheduleDecisionReason;
  admission.diagnostic =
      "source-backed production pressure profile and schedule decision accepted "
      "for selected-body resource-aware realization";
  return admission;
}

} // namespace

RVVLowPrecisionSameTargetMeasurementRecord
buildRVVPackedI4Gate4SameTargetMeasurementRecord(
    const RVVLowPrecisionContractionResourceSelection &selection) {
  RVVLowPrecisionSameTargetMeasurementRecord record;
  record.contract = kPackedI4MeasurementInputContract.str();
  record.authority = kPackedI4MeasurementInputAuthority.str();
  record.measurementEvidenceID =
      getRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceIDForCandidate(
          selection.selectedCandidateID)
          .str();
  record.measurementClassification =
      kPackedI4Gate4MeasurementClassification.str();
  record.measurementOutcomeFamily =
      kPackedI4Gate4MeasurementOutcomeFamily.str();
  record.measurementBestSpeedupRange =
      getPackedI4Gate4MeasurementBestSpeedupRangeForCandidate(
          selection.selectedCandidateID)
          .str();
  record.measurementSummaryRecordCount =
      getPackedI4Gate4MeasurementSummaryRecordCountForCandidate(
          selection.selectedCandidateID);
  record.measurementRecordCount =
      getPackedI4Gate4MeasurementRecordCountForCandidate(
          selection.selectedCandidateID);
  record.correctnessRecordCount =
      getPackedI4Gate4CorrectnessRecordCountForCandidate(
          selection.selectedCandidateID);
  record.sameTargetMeasurement = true;
  record.sshEvidence = true;
  record.targetProfile = kPackedI4Gate4TargetProfile.str();
  record.sourceRecordContract = kSourceBackedMeasurementRecordContract.str();
  record.sourceSelectedVariant =
      getPackedI4SourceSelectedVariantForCandidate(selection.selectedCandidateID)
          .str();
  record.sourceSelectedInput =
      getPackedI4SourceSelectedInputForCandidate(selection.selectedCandidateID)
          .str();
  record.sourceGeneratedFunction =
      getPackedI4SourceGeneratedFunctionForCandidate(
          selection.selectedCandidateID)
          .str();
  record.generatedArtifactIdentityContract =
      kGeneratedArtifactIdentityContract.str();
  record.generatedArtifactObjectPath =
      "generated-bundle-object:" + record.sourceGeneratedFunction;
  record.generatedArtifactObjectSHA256 =
      "source-backed-object-sha256:" + selection.selectedCandidateID;
  record.generatedArtifactHeaderPath =
      "generated-bundle-header:" + record.sourceGeneratedFunction;
  record.generatedArtifactHeaderSHA256 =
      "source-backed-header-sha256:" + selection.selectedCandidateID;
  record.measurementTarget = kPackedI4Gate4TargetProfile.str();
  record.measurementTargetProvenance = kMeasurementTargetProvenance.str();
  record.measurementRuntimeCountSet = "257,4096,65536";
  record.measurementRuntimeCountProvenance =
      kMeasurementRuntimeCountProvenance.str();
  record.pressureProfileLabel = kProductionPressureProfileLabel.str();
  record.pressureProfileLabelProvenance =
      kProductionPressureProfileLabelProvenance.str();
  record.providerResourceSelectedCandidate = selection.selectedCandidateID;
  record.providerResourcePlanningContract = selection.planningContract;
  record.providerResourceOperandForm = selection.operandForm;
  record.providerResourceSourceSignedness = selection.sourceSignedness;
  record.providerResourceStorageElementWidth = selection.storageElementWidth;
  record.providerResourceEffectiveElementWidth =
      selection.effectiveElementWidth;
  record.providerResourcePackingLayout = selection.packingLayout;
  record.providerResourceUnpackIntent = selection.unpackIntent;
  record.providerResourceVSetVLRegionCount = selection.vsetvlRegionCount;
  record.providerRuntimeAVLSource = selection.runtimeAVLSource;
  record.providerResourceRouteFamilyPlan = selection.routeFamilyPlanID;
  record.providerSupportedMirror = selection.providerSupportedMirror;
  record.providerRuntimeABIOrder = selection.runtimeABIOrder;
  record.providerScheduleDecisionContract =
      selection.scheduleDecisionContract;
  record.providerScheduleDecision = selection.scheduleDecision;
  record.providerScheduleDecisionReason =
      selection.scheduleDecisionReason;
  record.providerResourceCostContract = selection.resourceCostContract;
  record.providerResourceCostModel = selection.resourceCostModel;
  record.providerResourceCostLoopBodySteps =
      selection.resourceCostLoopBodySteps;
  record.providerResourceCostBlocker = selection.resourceCostBlocker;
  record.providerPerformanceAdmissionDecision =
      selection.performanceAdmissionDecision;
  record.providerPerformanceAdmissionClosure =
      selection.performanceAdmissionClosure;
  record.providerPerformanceAdmissionReopenRequirement =
      selection.performanceAdmissionReopenRequirement;
  record.providerBeyondLocalRepairAdmissionContract =
      selection.beyondLocalRepairAdmissionContract;
  record.providerBeyondLocalRepairAdmissionDecision =
      selection.beyondLocalRepairAdmissionDecision;
  record.providerBeyondLocalRepairAdmissionBlocker =
      selection.beyondLocalRepairAdmissionBlocker;
  record.providerBeyondLocalRepairAdmissionReopenRequirement =
      selection.beyondLocalRepairAdmissionReopenRequirement;
  record.providerRealizationAdmissionContract =
      selection.realizationAdmissionContract;
  record.providerRealizationAdmissionDecision =
      selection.realizationAdmissionDecision;
  record.providerRealizationAdmissionEvidence =
      selection.realizationAdmissionEvidence;
  record.providerRealizationAdmissionDispatchPolicy =
      selection.realizationAdmissionDispatchPolicy;
  record.providerRealizationAdmissionScheduleDecisionContract =
      selection.realizationAdmissionScheduleDecisionContract;
  record.providerRealizationAdmissionScheduleDecision =
      selection.realizationAdmissionScheduleDecision;
  record.providerRealizationAdmissionScheduleDecisionReason =
      selection.realizationAdmissionScheduleDecisionReason;
  record.providerPrimitiveChainContract =
      selection.primitiveChainContractID;
  record.providerPrimitiveChainKind = selection.primitiveChainKind;
  record.providerPrimitiveContract = selection.primitiveContractID;
  record.providerPrimitiveKind = selection.primitiveKind;
  record.providerWideningProductMultiplicandRoles =
      selection.wideningProductMultiplicandRoleSummary;
  record.providerWideningProductExtensionPolicy =
      selection.wideningProductExtensionPolicy;
  record.providerPrimitiveSourceLoad = selection.primitiveSourceLoadKind;
  record.providerPrimitiveSourceExtension =
      selection.primitiveSourceExtensionKind;
  record.providerPrimitiveSourceDType = selection.sourceElementTypeName;
  record.providerPrimitiveSourceSignedness = selection.sourceSignedness;
  record.providerPrimitiveSourceSEW = selection.sourceSEW;
  record.providerPrimitiveSourceLMUL = selection.sourceLMUL;
  record.providerPrimitiveProductDType = selection.productElementTypeName;
  record.providerPrimitiveProductSEW = selection.productSEW;
  record.providerPrimitiveProductLMUL = selection.productLMUL;
  record.providerPrimitiveAccumulatorDType =
      selection.accumulatorElementTypeName;
  record.providerPrimitiveAccumulatorSEW = selection.accumulatorSEW;
  record.providerPrimitiveAccumulatorLMUL = selection.accumulatorLMUL;
  record.providerPrimitiveResultDType = selection.resultElementTypeName;
  record.providerPrimitiveResultSEW = selection.resultSEW;
  record.providerPrimitiveResultLMUL = selection.resultLMUL;
  record.providerPrimitiveWideningProductRelation =
      selection.primitiveWideningProductRelation;
  record.providerPrimitiveProductReductionChainRelation =
      selection.primitiveProductReductionChainRelation;
  record.providerPrimitiveWideningProductIntrinsic =
      selection.primitiveWideningProductIntrinsic;
  record.providerPrimitiveReductionIntrinsic =
      selection.primitiveReductionIntrinsic;
  record.providerPrimitiveScalarSeedSplatIntrinsic =
      selection.primitiveScalarSeedSplatIntrinsic;
  record.providerPrimitiveAccumulatorLayout =
      selection.primitiveAccumulatorLayout;
  record.providerPrimitiveResultLayout = selection.primitiveResultLayout;
  record.providerPrimitiveReductionStoreVL =
      selection.primitiveReductionStoreVL;
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
  TCRV_READ_RECORD_STRING(sourceRecordContract, "source_record_contract");
  TCRV_READ_RECORD_STRING(sourceSelectedVariant, "source_selected_variant");
  TCRV_READ_RECORD_STRING(sourceSelectedInput, "source_selected_input");
  TCRV_READ_RECORD_STRING(sourceGeneratedFunction, "source_generated_function");
  TCRV_READ_RECORD_STRING(generatedArtifactIdentityContract,
                          "generated_artifact_identity_contract");
  TCRV_READ_RECORD_STRING(generatedArtifactObjectPath,
                          "generated_artifact_object_path");
  TCRV_READ_RECORD_STRING(generatedArtifactObjectSHA256,
                          "generated_artifact_object_sha256");
  TCRV_READ_RECORD_STRING(generatedArtifactHeaderPath,
                          "generated_artifact_header_path");
  TCRV_READ_RECORD_STRING(generatedArtifactHeaderSHA256,
                          "generated_artifact_header_sha256");
  TCRV_READ_RECORD_STRING(measurementTarget, "measurement_target");
  TCRV_READ_RECORD_STRING(measurementTargetProvenance,
                          "measurement_target_provenance");
  TCRV_READ_RECORD_STRING(measurementRuntimeCountSet,
                          "measurement_runtime_count_set");
  TCRV_READ_RECORD_STRING(measurementRuntimeCountProvenance,
                          "measurement_runtime_count_provenance");
  TCRV_READ_RECORD_STRING(pressureProfileLabel, "pressure_profile_label");
  TCRV_READ_RECORD_STRING(pressureProfileLabelProvenance,
                          "pressure_profile_label_provenance");
  TCRV_READ_RECORD_STRING(providerResourceSelectedCandidate,
                          "provider_resource_selected_candidate");
  TCRV_READ_RECORD_STRING(providerResourcePlanningContract,
                          "provider_resource_planning_contract");
  TCRV_READ_RECORD_STRING(providerResourceOperandForm,
                          "provider_resource_operand_form");
  TCRV_READ_RECORD_STRING(providerResourceSourceSignedness,
                          "provider_resource_source_signedness");
  TCRV_READ_RECORD_INT(providerResourceStorageElementWidth,
                       "provider_resource_storage_element_width");
  TCRV_READ_RECORD_INT(providerResourceEffectiveElementWidth,
                       "provider_resource_effective_element_width");
  TCRV_READ_RECORD_STRING(providerResourcePackingLayout,
                          "provider_resource_packing_layout");
  TCRV_READ_RECORD_STRING(providerResourceUnpackIntent,
                          "provider_resource_unpack_intent");
  TCRV_READ_RECORD_INT(providerResourceVSetVLRegionCount,
                       "provider_resource_vsetvl_region_count");
  TCRV_READ_RECORD_STRING(providerRuntimeAVLSource,
                          "provider_runtime_avl_source");
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
  TCRV_READ_RECORD_STRING(providerResourceCostContract,
                          "provider_resource_cost_contract");
  TCRV_READ_RECORD_STRING(providerResourceCostModel,
                          "provider_resource_cost_model");
  TCRV_READ_RECORD_INT(providerResourceCostLoopBodySteps,
                       "provider_resource_cost_loop_body_steps");
  TCRV_READ_RECORD_STRING(providerResourceCostBlocker,
                          "provider_resource_cost_blocker");
  TCRV_READ_RECORD_STRING(providerPerformanceAdmissionDecision,
                          "provider_performance_admission_decision");
  TCRV_READ_RECORD_STRING(providerPerformanceAdmissionClosure,
                          "provider_performance_admission_closure");
  TCRV_READ_RECORD_STRING(
      providerPerformanceAdmissionReopenRequirement,
      "provider_performance_admission_reopen_requirement");
  TCRV_READ_RECORD_STRING(
      providerBeyondLocalRepairAdmissionContract,
      "provider_beyond_local_repair_admission_contract");
  TCRV_READ_RECORD_STRING(
      providerBeyondLocalRepairAdmissionDecision,
      "provider_beyond_local_repair_admission_decision");
  TCRV_READ_RECORD_STRING(
      providerBeyondLocalRepairAdmissionBlocker,
      "provider_beyond_local_repair_admission_blocker");
  TCRV_READ_RECORD_STRING(
      providerBeyondLocalRepairAdmissionReopenRequirement,
      "provider_beyond_local_repair_admission_reopen_requirement");
  TCRV_READ_RECORD_STRING(providerRealizationAdmissionContract,
                          "provider_realization_admission_contract");
  TCRV_READ_RECORD_STRING(providerRealizationAdmissionDecision,
                          "provider_realization_admission_decision");
  TCRV_READ_RECORD_STRING(providerRealizationAdmissionEvidence,
                          "provider_realization_admission_evidence");
  TCRV_READ_RECORD_STRING(providerRealizationAdmissionDispatchPolicy,
                          "provider_realization_admission_dispatch_policy");
  TCRV_READ_RECORD_STRING(
      providerRealizationAdmissionScheduleDecisionContract,
      "provider_realization_admission_schedule_decision_contract");
  TCRV_READ_RECORD_STRING(
      providerRealizationAdmissionScheduleDecision,
      "provider_realization_admission_schedule_decision");
  TCRV_READ_RECORD_STRING(
      providerRealizationAdmissionScheduleDecisionReason,
      "provider_realization_admission_schedule_decision_reason");
  TCRV_READ_RECORD_STRING(providerPrimitiveChainContract,
                          "provider_primitive_chain_contract");
  TCRV_READ_RECORD_STRING(providerPrimitiveChainKind,
                          "provider_primitive_chain_kind");
  TCRV_READ_RECORD_STRING(providerPrimitiveContract,
                          "provider_primitive_contract");
  TCRV_READ_RECORD_STRING(providerPrimitiveKind, "provider_primitive_kind");
  TCRV_READ_RECORD_STRING(providerWideningProductMultiplicandRoles,
                          "provider_widening_product_multiplicand_roles");
  TCRV_READ_RECORD_STRING(providerWideningProductExtensionPolicy,
                          "provider_widening_product_extension_policy");
  TCRV_READ_RECORD_STRING(providerPrimitiveSourceLoad,
                          "provider_primitive_source_load");
  TCRV_READ_RECORD_STRING(providerPrimitiveSourceExtension,
                          "provider_primitive_source_extension");
  TCRV_READ_RECORD_STRING(providerPrimitiveSourceDType,
                          "provider_primitive_source_dtype");
  TCRV_READ_RECORD_STRING(providerPrimitiveSourceSignedness,
                          "provider_primitive_source_signedness");
  TCRV_READ_RECORD_INT(providerPrimitiveSourceSEW,
                       "provider_primitive_source_sew");
  TCRV_READ_RECORD_STRING(providerPrimitiveSourceLMUL,
                          "provider_primitive_source_lmul");
  TCRV_READ_RECORD_STRING(providerPrimitiveProductDType,
                          "provider_primitive_product_dtype");
  TCRV_READ_RECORD_INT(providerPrimitiveProductSEW,
                       "provider_primitive_product_sew");
  TCRV_READ_RECORD_STRING(providerPrimitiveProductLMUL,
                          "provider_primitive_product_lmul");
  TCRV_READ_RECORD_STRING(providerPrimitiveAccumulatorDType,
                          "provider_primitive_accumulator_dtype");
  TCRV_READ_RECORD_INT(providerPrimitiveAccumulatorSEW,
                       "provider_primitive_accumulator_sew");
  TCRV_READ_RECORD_STRING(providerPrimitiveAccumulatorLMUL,
                          "provider_primitive_accumulator_lmul");
  TCRV_READ_RECORD_STRING(providerPrimitiveResultDType,
                          "provider_primitive_result_dtype");
  TCRV_READ_RECORD_INT(providerPrimitiveResultSEW,
                       "provider_primitive_result_sew");
  TCRV_READ_RECORD_STRING(providerPrimitiveResultLMUL,
                          "provider_primitive_result_lmul");
  TCRV_READ_RECORD_STRING(providerPrimitiveWideningProductRelation,
                          "provider_primitive_widening_product_relation");
  TCRV_READ_RECORD_STRING(
      providerPrimitiveProductReductionChainRelation,
      "provider_primitive_product_reduction_chain_relation");
  TCRV_READ_RECORD_STRING(providerPrimitiveWideningProductIntrinsic,
                          "provider_primitive_widening_product_intrinsic");
  TCRV_READ_RECORD_STRING(providerPrimitiveReductionIntrinsic,
                          "provider_primitive_reduction_intrinsic");
  TCRV_READ_RECORD_STRING(providerPrimitiveScalarSeedSplatIntrinsic,
                          "provider_primitive_scalar_seed_splat_intrinsic");
  TCRV_READ_RECORD_STRING(providerPrimitiveAccumulatorLayout,
                          "provider_primitive_accumulator_layout");
  TCRV_READ_RECORD_STRING(providerPrimitiveResultLayout,
                          "provider_primitive_result_layout");
  TCRV_READ_RECORD_STRING(providerPrimitiveReductionStoreVL,
                          "provider_primitive_reduction_store_vl");
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

llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord>
buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceRoot(
    const llvm::json::Object &evidenceRoot, llvm::StringRef context) {
  llvm::Expected<const llvm::json::Object *> recordObject =
      requireEvidenceRootObject(evidenceRoot, context,
                                "same_target_measurement_record");
  if (!recordObject)
    return recordObject.takeError();

  llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord> record =
      buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceInput(
          **recordObject, context);
  if (!record)
    return record.takeError();
  if (llvm::Error error =
          verifyPackedI4SameTargetEvidenceRoot(evidenceRoot, *record, context))
    return std::move(error);
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
  input.providerResourcePlanningContract = selection.planningContract;
  input.providerResourceOperandForm = selection.operandForm;
  input.providerResourceSourceSignedness = selection.sourceSignedness;
  input.providerResourceStorageElementWidth = selection.storageElementWidth;
  input.providerResourceEffectiveElementWidth = selection.effectiveElementWidth;
  input.providerResourcePackingLayout = selection.packingLayout;
  input.providerResourceUnpackIntent = selection.unpackIntent;
  input.providerResourceVSetVLRegionCount = selection.vsetvlRegionCount;
  input.providerRuntimeAVLSource = selection.runtimeAVLSource;
  input.providerResourceRouteFamilyPlan = selection.routeFamilyPlanID;
  input.providerSupportedMirror = selection.providerSupportedMirror;
  input.providerRuntimeABIOrder = selection.runtimeABIOrder;
  input.providerScheduleDecisionContract =
      outcome.providerScheduleDecisionContract;
  input.providerScheduleDecision = outcome.providerScheduleDecision;
  input.providerScheduleDecisionReason =
      outcome.providerScheduleDecisionReason;
  input.providerResourceCostContract = selection.resourceCostContract;
  input.providerResourceCostModel = selection.resourceCostModel;
  input.providerResourceCostLoopBodySteps =
      selection.resourceCostLoopBodySteps;
  input.providerResourceCostBlocker = selection.resourceCostBlocker;
  input.providerPerformanceAdmissionDecision =
      selection.performanceAdmissionDecision;
  input.providerPerformanceAdmissionClosure =
      selection.performanceAdmissionClosure;
  input.providerPerformanceAdmissionReopenRequirement =
      selection.performanceAdmissionReopenRequirement;
  input.providerBeyondLocalRepairAdmissionContract =
      selection.beyondLocalRepairAdmissionContract;
  input.providerBeyondLocalRepairAdmissionDecision =
      selection.beyondLocalRepairAdmissionDecision;
  input.providerBeyondLocalRepairAdmissionBlocker =
      selection.beyondLocalRepairAdmissionBlocker;
  input.providerBeyondLocalRepairAdmissionReopenRequirement =
      selection.beyondLocalRepairAdmissionReopenRequirement;
  input.providerRealizationAdmissionContract =
      selection.realizationAdmissionContract;
  input.providerRealizationAdmissionDecision =
      selection.realizationAdmissionDecision;
  input.providerRealizationAdmissionEvidence =
      selection.realizationAdmissionEvidence;
  input.providerRealizationAdmissionDispatchPolicy =
      selection.realizationAdmissionDispatchPolicy;
  input.providerRealizationAdmissionScheduleDecisionContract =
      selection.realizationAdmissionScheduleDecisionContract;
  input.providerRealizationAdmissionScheduleDecision =
      selection.realizationAdmissionScheduleDecision;
  input.providerRealizationAdmissionScheduleDecisionReason =
      selection.realizationAdmissionScheduleDecisionReason;
  input.providerPrimitiveChainContract = selection.primitiveChainContractID;
  input.providerPrimitiveChainKind = selection.primitiveChainKind;
  input.providerPrimitiveContract = selection.primitiveContractID;
  input.providerPrimitiveKind = selection.primitiveKind;
  input.providerWideningProductMultiplicandRoles =
      selection.wideningProductMultiplicandRoleSummary;
  input.providerWideningProductExtensionPolicy =
      selection.wideningProductExtensionPolicy;
  input.providerPrimitiveSourceLoad = selection.primitiveSourceLoadKind;
  input.providerPrimitiveSourceExtension =
      selection.primitiveSourceExtensionKind;
  input.providerPrimitiveSourceDType = selection.sourceElementTypeName;
  input.providerPrimitiveSourceSignedness = selection.sourceSignedness;
  input.providerPrimitiveSourceSEW = selection.sourceSEW;
  input.providerPrimitiveSourceLMUL = selection.sourceLMUL;
  input.providerPrimitiveProductDType = selection.productElementTypeName;
  input.providerPrimitiveProductSEW = selection.productSEW;
  input.providerPrimitiveProductLMUL = selection.productLMUL;
  input.providerPrimitiveAccumulatorDType =
      selection.accumulatorElementTypeName;
  input.providerPrimitiveAccumulatorSEW = selection.accumulatorSEW;
  input.providerPrimitiveAccumulatorLMUL = selection.accumulatorLMUL;
  input.providerPrimitiveResultDType = selection.resultElementTypeName;
  input.providerPrimitiveResultSEW = selection.resultSEW;
  input.providerPrimitiveResultLMUL = selection.resultLMUL;
  input.providerPrimitiveWideningProductRelation =
      selection.primitiveWideningProductRelation;
  input.providerPrimitiveProductReductionChainRelation =
      selection.primitiveProductReductionChainRelation;
  input.providerPrimitiveWideningProductIntrinsic =
      selection.primitiveWideningProductIntrinsic;
  input.providerPrimitiveReductionIntrinsic =
      selection.primitiveReductionIntrinsic;
  input.providerPrimitiveScalarSeedSplatIntrinsic =
      selection.primitiveScalarSeedSplatIntrinsic;
  input.providerPrimitiveAccumulatorLayout =
      selection.primitiveAccumulatorLayout;
  input.providerPrimitiveResultLayout = selection.primitiveResultLayout;
  input.providerPrimitiveReductionStoreVL =
      selection.primitiveReductionStoreVL;

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
  if (hasPackedI4SiblingRouteMeasurement(
          selection, input.providerResourceSelectedCandidate))
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

llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput>
buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceRoot(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot, llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementRecord> record =
      buildRVVLowPrecisionSameTargetMeasurementRecordFromEvidenceRoot(
          evidenceRoot, context);
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

llvm::Expected<RVVLowPrecisionProductionPressureProfile>
buildRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  if (llvm::Error error = rejectPressureProfileMarkerOnlyFacts(
          selection, input, dispatchBoundary, context))
    return std::move(error);
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, input, context);
  if (!decision)
    return decision.takeError();
  return materializeRVVLowPrecisionProductionPressureProfile(
      selection, input, dispatchBoundary, *decision, context);
}

llvm::Expected<RVVLowPrecisionProductionPressureProfile>
buildRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> input =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInput(selection, record,
                                                           context);
  if (!input)
    return input.takeError();
  return buildRVVLowPrecisionProductionPressureProfile(selection, *input,
                                                      dispatchBoundary,
                                                      context);
}

llvm::Error populateRVVLowPrecisionSelectedBodyRealizationAdmissionProof(
    RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  if (!isRVVLowPrecisionResourcePackedI4CandidateID(
          selection.selectedCandidateID))
    return llvm::Error::success();

  auto setOrVerify = [&](llvm::StringRef label, std::string &field,
                         llvm::StringRef expected) -> llvm::Error {
    if (field.empty()) {
      field = expected.str();
      return llvm::Error::success();
    }
    return requirePolicyString(context, label, field, expected);
  };

  RVVLowPrecisionSameTargetMeasurementRecord record =
      buildRVVPackedI4Gate4SameTargetMeasurementRecord(selection);
  const std::string measurementEvidenceID =
      selection.remediationMeasurementEvidenceID.empty()
          ? record.measurementEvidenceID
          : selection.remediationMeasurementEvidenceID;
  if (llvm::Error error = setOrVerify(
          "realization admission contract",
          selection.realizationAdmissionContract,
          kSelectedBodyRealizationAdmissionContract))
    return error;
  if (llvm::Error error = setOrVerify(
          "realization admission decision",
          selection.realizationAdmissionDecision,
          stringifyRVVLowPrecisionRealizationAdmissionDecision(
              RVVLowPrecisionRealizationAdmissionDecision::Realize)))
    return error;
  if (llvm::Error error =
          setOrVerify("realization admission evidence",
                      selection.realizationAdmissionEvidence,
                      measurementEvidenceID))
    return error;
  if (selection.realizationAdmissionDispatchPolicy.empty()) {
    selection.realizationAdmissionDispatchPolicy =
        selection.dispatchPreference == kPackedI4PerformancePreferredPolicyPath
            ? kPackedI4PerformancePreferredPolicyPath.str()
            : kPackedI4CorrectnessFallbackPolicyPath.str();
  }
  if (llvm::Error error = setOrVerify(
          "realization admission schedule decision contract",
          selection.realizationAdmissionScheduleDecisionContract,
          selection.scheduleDecisionContract))
    return error;
  if (llvm::Error error = setOrVerify(
          "realization admission schedule decision",
          selection.realizationAdmissionScheduleDecision,
          selection.scheduleDecision))
    return error;
  if (llvm::Error error = setOrVerify(
          "realization admission schedule decision reason",
          selection.realizationAdmissionScheduleDecisionReason,
          selection.scheduleDecisionReason))
    return error;

  if (!dispatchBoundary.hasFacts())
    return llvm::Error::success();

  record = buildRVVPackedI4Gate4SameTargetMeasurementRecord(selection);
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, record,
                                               dispatchBoundary, context);
  if (!decision)
    return decision.takeError();
  if (llvm::Error error =
          setOrVerify("realization admission dispatch policy",
                      selection.realizationAdmissionDispatchPolicy,
                      decision->dispatchPolicyPath))
    return error;

  record = buildRVVPackedI4Gate4SameTargetMeasurementRecord(selection);
  llvm::Expected<RVVLowPrecisionProductionPressureProfile> profile =
      buildRVVLowPrecisionProductionPressureProfile(selection, record,
                                                   dispatchBoundary, context);
  if (!profile)
    return profile.takeError();
  llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission> admission =
      admitRVVLowPrecisionSelectedBodyRealization(selection, &*profile,
                                                  context);
  if (!admission)
    return admission.takeError();
  if (llvm::Error error = requirePolicyString(
          context, "realization admission contract",
          selection.realizationAdmissionContract,
          admission->admissionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission decision",
          selection.realizationAdmissionDecision,
          stringifyRVVLowPrecisionRealizationAdmissionDecision(
              admission->decision)))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission evidence",
          selection.realizationAdmissionEvidence,
          admission->measurementEvidenceID))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission dispatch policy",
          selection.realizationAdmissionDispatchPolicy,
          admission->dispatchPolicyPath))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission schedule decision contract",
          selection.realizationAdmissionScheduleDecisionContract,
          admission->scheduleDecisionContract))
    return error;
  if (llvm::Error error = requirePolicyString(
          context, "realization admission schedule decision",
          selection.realizationAdmissionScheduleDecision,
          admission->scheduleDecision))
    return error;
  return requirePolicyString(
      context, "realization admission schedule decision reason",
      selection.realizationAdmissionScheduleDecisionReason,
      admission->scheduleDecisionReason);
}

llvm::Error verifyRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementPolicyInput &input,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionProductionPressureProfile> profile =
      buildRVVLowPrecisionProductionPressureProfile(selection, input,
                                                   dispatchBoundary, context);
  if (!profile)
    return profile.takeError();
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionProductionPressureProfile(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionSameTargetMeasurementRecord &record,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionProductionPressureProfile> profile =
      buildRVVLowPrecisionProductionPressureProfile(selection, record,
                                                   dispatchBoundary, context);
  if (!profile)
    return profile.takeError();
  return llvm::Error::success();
}

bool RVVLowPrecisionSelectedBodyRealizationAdmission::admitsRealization()
    const {
  return decision == RVVLowPrecisionRealizationAdmissionDecision::Realize;
}

llvm::StringRef stringifyRVVLowPrecisionRealizationAdmissionDecision(
    RVVLowPrecisionRealizationAdmissionDecision decision) {
  switch (decision) {
  case RVVLowPrecisionRealizationAdmissionDecision::Realize:
    return "realize";
  case RVVLowPrecisionRealizationAdmissionDecision::Defer:
    return "defer";
  case RVVLowPrecisionRealizationAdmissionDecision::Deny:
    return "deny";
  }
  return "unknown";
}

llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission>
admitRVVLowPrecisionSelectedBodyRealization(
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context) {
  if (!profile)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-body realization admission requires source-backed "
        "production pressure profile before resource-aware realization");
  if (llvm::Error error =
          verifyProductionPressureProfileAgainstCandidate(candidate, *profile,
                                                         context))
    return std::move(error);
  return makeSelectedBodyRealizationAdmission(*profile);
}

llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission>
admitRVVLowPrecisionSelectedBodyRealization(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context) {
  if (!profile)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-body realization admission requires source-backed "
        "production pressure profile before resource-aware realization");
  if (llvm::Error error =
          verifyProductionPressureProfileAgainstSelection(selection, *profile,
                                                         context))
    return std::move(error);
  return makeSelectedBodyRealizationAdmission(*profile);
}

llvm::Error verifyRVVLowPrecisionSelectedBodyRealizationAdmission(
    const RVVLowPrecisionContractionResourceCandidate &candidate,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission> admission =
      admitRVVLowPrecisionSelectedBodyRealization(candidate, profile, context);
  if (!admission)
    return admission.takeError();
  return llvm::Error::success();
}

llvm::Error verifyRVVLowPrecisionSelectedBodyRealizationAdmission(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const RVVLowPrecisionProductionPressureProfile *profile,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSelectedBodyRealizationAdmission> admission =
      admitRVVLowPrecisionSelectedBodyRealization(selection, profile, context);
  if (!admission)
    return admission.takeError();
  return llvm::Error::success();
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
  handoff.expectedSelectedCandidateID = selection.selectedCandidateID;
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
  if (hasPackedI4SiblingRouteMeasurement(
          selection, input.providerResourceSelectedCandidate)) {
    RVVLowPrecisionPerformancePolicyHandoff handoff;
    handoff.handoffContract =
        kRVVLowPrecisionResourcePackedI4RemediationHandoffContract.str();
    handoff.selectedCandidateID = selection.selectedCandidateID;
    handoff.expectedSelectedCandidateID = selection.selectedCandidateID;
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
         " diagnosed stale sibling-route measurement: measurement provider "
         "selected candidate '" +
         input.providerResourceSelectedCandidate +
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
  handoff.expectedSelectedCandidateID = selection.selectedCandidateID;
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
  handoff.expectedSelectedCandidateID = selection.selectedCandidateID;
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
  llvm::Expected<RVVLowPrecisionProductionPressureProfile> pressureProfile =
      materializeRVVLowPrecisionProductionPressureProfile(
          selection, input, dispatchBoundary, *decision, context);
  if (!pressureProfile)
    return pressureProfile.takeError();
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
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> input =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInput(selection, record,
                                                           context);
  if (!input)
    return input.takeError();
  llvm::Expected<RVVLowPrecisionProductionPressureProfile> pressureProfile =
      materializeRVVLowPrecisionProductionPressureProfile(
          selection, *input, dispatchBoundary, *decision, context);
  if (!pressureProfile)
    return pressureProfile.takeError();
  return decision;
}

llvm::Expected<RVVLowPrecisionPerformancePolicyDecision>
evaluateRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> input =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInputFromEvidenceRoot(
          selection, evidenceRoot, context);
  if (!input)
    return input.takeError();
  return evaluateRVVLowPrecisionPerformancePolicy(selection, *input,
                                                 dispatchBoundary, context);
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
    return decision;
  }
  llvm::Expected<RVVLowPrecisionSameTargetMeasurementPolicyInput> input =
      buildRVVLowPrecisionSameTargetMeasurementPolicyInput(selection, record,
                                                           context);
  if (!input) {
    decision.performanceSelectionAllowed = false;
    decision.performanceWinClaimAllowed = false;
    decision.performancePreferredPathSelected = false;
    decision.dispatchPreference =
        kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
    decision.performancePreferenceDenialReason =
        llvm::toString(input.takeError());
    populateRVVLowPrecisionPolicyDispatchPath(decision);
    return decision;
  }
  llvm::Expected<RVVLowPrecisionProductionPressureProfile> pressureProfile =
      materializeRVVLowPrecisionProductionPressureProfile(
          selection, *input, dispatchBoundary, decision, context);
  if (!pressureProfile) {
    decision.performanceSelectionAllowed = false;
    decision.performanceWinClaimAllowed = false;
    decision.performancePreferredPathSelected = false;
    decision.dispatchPreference =
        kRVVLowPrecisionResourcePackedI4DispatchPreference.str();
    decision.performancePreferenceDenialReason =
        llvm::toString(pressureProfile.takeError());
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

llvm::Error verifyRVVLowPrecisionPerformancePolicy(
    const RVVLowPrecisionContractionResourceSelection &selection,
    const llvm::json::Object &evidenceRoot,
    const RVVLowPrecisionSelectedDispatchPolicyBoundary &dispatchBoundary,
    llvm::StringRef context) {
  llvm::Expected<RVVLowPrecisionPerformancePolicyDecision> decision =
      evaluateRVVLowPrecisionPerformancePolicy(selection, evidenceRoot,
                                               dispatchBoundary, context);
  if (!decision)
    return decision.takeError();
  if (!decision->routeSupportAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve executable route support for the accepted selected-"
        "dispatch same-target measurement evidence root");
  if (!decision->correctnessExecutionAllowed)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " must preserve correctness execution for the accepted selected-"
        "dispatch same-target measurement evidence root");
  if (decision->performancePreferredPathSelected)
    return llvm::Error::success();
  if (decision->performanceSelectionAllowed ||
      decision->performanceWinClaimAllowed ||
      !decision->correctnessFallbackPathSelected ||
      decision->dispatchPolicyPath != kPackedI4CorrectnessFallbackPolicyPath)
    return makeRVVLowPrecisionPerformancePolicyError(
        llvm::Twine(context) +
        " selected-dispatch same-target measurement evidence root must deny "
        "performance selection and preserve the conservative correctness "
        "fallback path");
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
