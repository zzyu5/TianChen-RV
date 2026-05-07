#ifndef TIANCHENRV_DIALECT_EXEC_IR_DIAGNOSTICCONVENTIONS_H
#define TIANCHENRV_DIALECT_EXEC_IR_DIAGNOSTICCONVENTIONS_H

#include "llvm/ADT/StringRef.h"

namespace tianchenrv::tcrv::exec::diagnostic {

inline constexpr llvm::StringLiteral kArtifactKindAttrName("artifact_kind");
inline constexpr llvm::StringLiteral kEmissionKindAttrName("emission_kind");
inline constexpr llvm::StringLiteral kLoweringPipelineAttrName(
    "lowering_pipeline");
inline constexpr llvm::StringLiteral kLoweringBoundaryAttrName(
    "lowering_boundary");
inline constexpr llvm::StringLiteral kMessageAttrName("message");
inline constexpr llvm::StringLiteral kOriginAttrName("origin");
inline constexpr llvm::StringLiteral kPlanKindAttrName("plan_kind");
inline constexpr llvm::StringLiteral kReasonAttrName("reason");
inline constexpr llvm::StringLiteral kRoleAttrName("role");
inline constexpr llvm::StringLiteral kRuntimeABIAttrName("runtime_abi");
inline constexpr llvm::StringLiteral kRuntimeABIKindAttrName(
    "runtime_abi_kind");
inline constexpr llvm::StringLiteral kRuntimeABINameAttrName(
    "runtime_abi_name");
inline constexpr llvm::StringLiteral kRuntimeABIParametersAttrName(
    "runtime_abi_parameters");
inline constexpr llvm::StringLiteral kRuntimeGlueRoleAttrName(
    "runtime_glue_role");
inline constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
inline constexpr llvm::StringLiteral kSelectionKindAttrName("selection_kind");
inline constexpr llvm::StringLiteral kSeverityAttrName("severity");
inline constexpr llvm::StringLiteral kStatusAttrName("status");
inline constexpr llvm::StringLiteral kTargetAttrName("target");

inline constexpr llvm::StringLiteral kEmissionPlanReasonValue("emission_plan");
inline constexpr llvm::StringLiteral kEmissionPlanPlanKindValue(
    "plugin-emission-plan");
inline constexpr llvm::StringLiteral kEmissionPlanSupportedStatusValue(
    "supported");
inline constexpr llvm::StringLiteral kEmissionPlanUnsupportedStatusValue(
    "unsupported");
inline constexpr llvm::StringLiteral kEmissionPlanMetadataOnlyStatusValue(
    "metadata-only");
inline constexpr llvm::StringLiteral kEmissionPlanSupportedSeverityValue(
    "info");
inline constexpr llvm::StringLiteral kEmissionPlanUnsupportedSeverityValue(
    "error");
inline constexpr llvm::StringLiteral kEmissionPlanMetadataOnlySeverityValue(
    "note");

inline constexpr llvm::StringLiteral kSelectedReasonValue("variant-selected");
inline constexpr llvm::StringLiteral kStaticSelectionKindValue(
    "static-variant");
inline constexpr llvm::StringLiteral kFallbackOnlySelectionKindValue(
    "fallback-only");

inline bool isEmissionPlanReason(llvm::StringRef reason) {
  return reason == kEmissionPlanReasonValue;
}

inline bool isEmissionPlanStatus(llvm::StringRef status) {
  return status == kEmissionPlanSupportedStatusValue ||
         status == kEmissionPlanMetadataOnlyStatusValue ||
         status == kEmissionPlanUnsupportedStatusValue;
}

} // namespace tianchenrv::tcrv::exec::diagnostic

#endif // TIANCHENRV_DIALECT_EXEC_IR_DIAGNOSTICCONVENTIONS_H
