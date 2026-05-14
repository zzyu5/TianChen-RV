#ifndef TIANCHENRV_TARGET_RVV_RVVSELECTEDCONFIGCONTRACT_H
#define TIANCHENRV_TARGET_RVV_RVVSELECTEDCONFIGCONTRACT_H

#include "TianChenRV/Support/FiniteBinaryFrontendLowering.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamily.h"
#include "TianChenRV/Target/RVV/RVVRuntimeLengthContract.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::target::rvv {

inline llvm::StringRef getRVVSelectedBinaryDTypeMetadataName() {
  return "tcrv_rvv.selected_binary_dtype";
}

inline llvm::StringRef getRVVSelectedBinaryFamilyMetadataName() {
  return "tcrv_rvv.selected_binary_family";
}

inline llvm::StringRef getRVVSelectedBinaryOperatorMetadataName() {
  return "tcrv_rvv.selected_binary_operator";
}

inline llvm::StringRef getRVVSelectedBinarySourceKindMetadataName() {
  return "tcrv_rvv.selected_binary_source_kind";
}

inline llvm::StringRef getRVVSelectedBinaryMicrokernelOpMetadataName() {
  return "tcrv_rvv.selected_binary_microkernel_op";
}

inline llvm::StringRef getRVVEmitCSourceOpMetadataName() {
  return "tcrv_rvv.emitc_source_op";
}

inline llvm::StringRef getRVVEmitCLowerableOpInterfaceMetadataName() {
  return "tcrv_rvv.emitc_lowerable_op_interface";
}

inline llvm::StringRef getRVVEmitCRouteKindMetadataName() {
  return "tcrv_rvv.emitc_route_kind";
}

inline llvm::StringRef getRVVEmitCSourceAuthorityMetadataName() {
  return "tcrv_rvv.emitc_source_authority";
}

inline llvm::StringRef getRVVEmitCRequiredHeaderMetadataName() {
  return "tcrv_rvv.emitc_required_header";
}

inline llvm::StringRef getRVVEmitCRouteKindMetadataValue() {
  return "extension-family-ops-to-emitc-call-opaque";
}

inline llvm::StringRef getRVVEmitCSourceAuthorityMetadataValue() {
  return "mlir-emitc-cpp-emitter";
}

inline llvm::StringRef getRVVEmitCRequiredHeaderMetadataValue() {
  return "riscv_vector.h";
}

inline llvm::StringRef getRVVTypedBinarySourceMetadataRole() {
  return "typed-rvv-binary-source";
}

inline llvm::StringRef getRVVTypedBinarySourceMetadataNote() {
  return "typed RVV family-op metadata selected by the RVV plugin for the "
         "common EmitC route; not metadata-owned computation, runtime "
         "correctness evidence, or performance evidence";
}

inline llvm::StringRef getRVVTypedBinarySourceIdentityMetadataRole() {
  return "typed-rvv-binary-source-identity";
}

inline llvm::StringRef getRVVTypedBinarySourceIdentityMetadataNote() {
  return "op-owned RVV source identity carried from the selected lowering "
         "boundary into selected-plan and artifact-bundle validation; not "
         "metadata-owned computation or runtime evidence";
}

inline llvm::StringRef getRVVEmitCSourceOpMetadataRole() {
  return "typed-rvv-emitc-source-op";
}

inline llvm::StringRef getRVVEmitCSourceOpMetadataNote() {
  return "typed RVV arithmetic op used as the source operation for the common "
         "EmitC lowerable route; not a legacy route label";
}

inline llvm::StringRef getRVVEmitCLowerableOpInterfaceMetadataNote() {
  return "generated RVV op interface queried before building the common EmitC "
         "lowerable route; not metadata-selected computation";
}

inline llvm::StringRef getRVVEmitCRouteMetadataRole() {
  return "typed-rvv-emitc-route";
}

inline llvm::StringRef getRVVEmitCRouteMetadataNote() {
  return "RVV plugin-selected extension-family op to common EmitC route "
         "mapping consumed before target artifact export; not legacy direct C "
         "emission";
}

inline llvm::StringRef getRVVSelectedConfigProfileHardwareFactsMetadataName() {
  return "tcrv_rvv.selected_config_profile.hardware_facts";
}

inline llvm::StringRef getRVVSelectedConfigProfileVariantConfigMetadataName() {
  return "tcrv_rvv.selected_config_profile.variant_config";
}

inline llvm::StringRef getRVVSelectedConfigProfileRuntimeRolesMetadataName() {
  return "tcrv_rvv.selected_config_profile.runtime_roles";
}

inline llvm::StringRef getRVVSelectedConfigProfileMetadataRole() {
  return "rvv-selected-config-profile";
}

inline llvm::StringRef getRVVSelectedConfigProfileMetadataNote() {
  return "plugin-owned RVV selected config profile separating target "
         "capability facts, compile-time vector variant config, and runtime "
         "AVL/VL roles; not metadata-owned computation or runtime evidence";
}

class RVVBinarySelectedConfigContract {
public:
  RVVBinarySelectedConfigContract() = default;

  RVVBinarySelectedConfigContract(const RVVBinaryFamilyRecord &family,
                                  const RVVVectorShapeConfig &shape)
      : family(&family), shape(&shape) {}

  bool isValid() const { return family && shape; }

  const RVVBinaryFamilyRecord &getFamily() const { return *family; }
  const RVVVectorShapeConfig &getShape() const { return *shape; }

  llvm::StringRef getDTypeID() const {
    return family ? family->dtypeID : llvm::StringRef();
  }
  llvm::StringRef getFamilyID() const {
    return family ? family->familyID : llvm::StringRef();
  }
  llvm::StringRef getFrontendLowering() const {
    return family ? family->frontendLowering : llvm::StringRef();
  }
  llvm::StringRef getArithmeticOpName() const {
    return family ? family->arithmeticOpName : llvm::StringRef();
  }
  llvm::StringRef getArithmeticVerb() const {
    return family ? family->arithmeticVerb : llvm::StringRef();
  }
  llvm::StringRef getConstInputPointerCType() const {
    return family ? family->constInputPointerCType : llvm::StringRef();
  }
  llvm::StringRef getOutputPointerCType() const {
    return family ? family->outputPointerCType : llvm::StringRef();
  }

  llvm::StringRef getShapeID() const {
    return shape ? shape->shapeID : llvm::StringRef();
  }
  std::int64_t getSEWBits() const { return shape ? shape->sewBits : 0; }
  llvm::StringRef getLMUL() const {
    return shape ? shape->lmul : llvm::StringRef();
  }
  llvm::StringRef getTailPolicy() const {
    return shape ? shape->tailPolicy : llvm::StringRef();
  }
  llvm::StringRef getMaskPolicy() const {
    return shape ? shape->maskPolicy : llvm::StringRef();
  }
  llvm::StringRef getVectorType() const {
    return shape ? shape->vectorType : llvm::StringRef();
  }
  llvm::StringRef getVectorSuffix() const {
    return shape ? shape->vectorSuffix : llvm::StringRef();
  }
  llvm::StringRef getSetVLSuffix() const {
    return shape ? shape->setvlSuffix : llvm::StringRef();
  }

  llvm::StringRef getSelectedVariantSymbol() const {
    return selectedVariantSymbol;
  }
  llvm::StringRef getSelectedRole() const { return selectedRole; }
  std::int64_t getComponentCapacityElementCount() const {
    return runtimeLength.getComponentCapacityElementCount();
  }
  llvm::StringRef getRuntimeElementCountCName() const {
    return runtimeLength.getRuntimeElementCountCName();
  }
  llvm::StringRef getDispatchAvailabilityGuardCName() const {
    return dispatchAvailabilityGuardCName;
  }
  llvm::StringRef getRuntimeAVLSource() const {
    return runtimeLength.getRuntimeAVLSource();
  }
  llvm::StringRef getRuntimeAVLRole() const {
    return runtimeLength.getRuntimeAVLRole();
  }
  llvm::StringRef getRuntimeVLSource() const {
    return runtimeLength.getRuntimeVLSource();
  }
  llvm::StringRef getRuntimeVLScope() const {
    return runtimeLength.getRuntimeVLScope();
  }
  const RVVRuntimeLengthContract &getRuntimeLengthContract() const {
    return runtimeLength;
  }
  const std::optional<support::FixedVectorSourceExtentContract> &
  getFixedVectorSourceExtentContract() const {
    return fixedSourceExtent;
  }
  const std::optional<support::DynamicVectorRuntimeExtentContract> &
  getDynamicVectorRuntimeExtentContract() const {
    return dynamicRuntimeExtent;
  }

  void setSelectedPath(llvm::StringRef variantSymbol, llvm::StringRef role) {
    selectedVariantSymbol = variantSymbol.trim().str();
    selectedRole = role.trim().str();
  }

  void setComponentCapacityElementCount(std::int64_t count) {
    runtimeLength.setComponentCapacityElementCount(count);
  }

  void setRuntimeElementCountCName(llvm::StringRef cName) {
    runtimeLength.setRuntimeElementCountCName(cName);
  }

  void setDispatchAvailabilityGuardCName(llvm::StringRef cName) {
    llvm::StringRef trimmed = cName.trim();
    dispatchAvailabilityGuardCName =
        trimmed.empty() ? "rvv_available" : trimmed.str();
  }

  void setFixedVectorSourceExtentContract(
      std::optional<support::FixedVectorSourceExtentContract> contract) {
    fixedSourceExtent = std::move(contract);
  }

  void setDynamicVectorRuntimeExtentContract(
      std::optional<support::DynamicVectorRuntimeExtentContract> contract) {
    dynamicRuntimeExtent = std::move(contract);
  }

  llvm::SmallVector<llvm::StringRef, 4> getSelectedShapeCapabilityIDs() const {
    llvm::SmallVector<llvm::StringRef, 4> ids;
    if (!shape)
      return ids;
    ids.push_back(shape->sewCapabilityID);
    ids.push_back(shape->lmulCapabilityID);
    ids.push_back(shape->tailPolicyCapabilityID);
    ids.push_back(shape->maskPolicyCapabilityID);
    return ids;
  }

  std::string formatSelectedVectorShapeConfigCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_vector_shape_config: ";
    if (getDTypeID() != "i32")
      stream << "dtype=" << getDTypeID() << ", ";
    stream << "shape=" << getShapeID() << ", sew=" << getSEWBits()
           << ", lmul=" << getLMUL()
           << ", tail_policy=" << getTailPolicy()
           << ", mask_policy=" << getMaskPolicy()
           << ", vector_type=" << getVectorType()
           << ", vector_suffix=" << getVectorSuffix()
           << ", setvl_suffix=" << getSetVLSuffix();
    stream.flush();
    return text;
  }

  std::string formatSelectedVectorShapeCapabilitiesCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_vector_shape_capabilities:";
    for (llvm::StringRef capabilityID : getSelectedShapeCapabilityIDs())
      stream << " " << capabilityID;
    stream.flush();
    return text;
  }

  std::string formatSelectedConfigProfileHardwareFactsMetadataValue() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "hw=target-capability-profile"
           << ",shape=" << getShapeID() << ",caps=";
    llvm::SmallVector<llvm::StringRef, 4> capabilityIDs =
        getSelectedShapeCapabilityIDs();
    bool first = true;
    for (llvm::StringRef capabilityID : capabilityIDs) {
      if (!first)
        stream << "|";
      first = false;
      stream << capabilityID;
    }
    stream.flush();
    return text;
  }

  std::string formatSelectedConfigProfileVariantConfigMetadataValue() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "variant=rvv-plugin-selected-vector-config"
           << ",dtype=" << getDTypeID() << ",family=" << getFamilyID()
           << ",op=" << getArithmeticVerb()
           << ",shape=" << getShapeID() << ",sew=" << getSEWBits()
           << ",lmul=" << getLMUL()
           << ",tail=" << getTailPolicy()
           << ",mask=" << getMaskPolicy()
           << ",vtype=" << getVectorType()
           << ",vsuffix=" << getVectorSuffix()
           << ",setvl=" << getSetVLSuffix();
    stream.flush();
    return text;
  }

  std::string formatSelectedConfigProfileRuntimeRolesMetadataValue() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "runtime=runtime-abi-ssa-control"
           << ",n="
           << getRuntimeElementCountCName()
           << ",avl=" << getRuntimeAVLSource() << "/" << getRuntimeAVLRole()
           << ",vl=" << getRuntimeVLSource() << "/" << getRuntimeVLScope();
    if (fixedSourceExtent)
      stream << ",fixed_extent="
             << fixedSourceExtent->sourceVectorExtent
             << ",constraint="
             << fixedSourceExtent->runtimeElementCountConstraint;
    if (dynamicRuntimeExtent)
      stream << ",dynamic_extent_arg="
             << dynamicRuntimeExtent->runtimeExtentArg
             << ",loop_step=" << dynamicRuntimeExtent->sourceLoopStep
             << ",chunk="
             << dynamicRuntimeExtent->sourceVectorChunkExtent
             << ",active_lanes=source-frontdoor"
             << ",tail=runtime-bounded"
             << ",constraint="
             << dynamicRuntimeExtent->runtimeElementCountConstraint;
    stream.flush();
    return text;
  }

  std::string formatSelectedConfigProfileCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_config_profile: hardware_facts={"
           << formatSelectedConfigProfileHardwareFactsMetadataValue()
           << "}, variant_config={"
           << formatSelectedConfigProfileVariantConfigMetadataValue()
           << "}, runtime_roles={"
           << formatSelectedConfigProfileRuntimeRolesMetadataValue()
           << "}, component_capacity_element_count=" << getComponentCapacityElementCount()
           << ", source=RVVBinarySelectedConfigProfile";
    stream.flush();
    return text;
  }

  std::string formatSelectedConfigEmissionAuthorityCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_config_emission_authority: vector_type="
           << getVectorType() << ", vector_suffix=" << getVectorSuffix()
           << ", setvl_suffix=" << getSetVLSuffix()
           << ", tail_policy=" << getTailPolicy()
           << ", mask_policy=" << getMaskPolicy()
           << ", source=RVVBinarySelectedConfigContract";
    stream.flush();
    return text;
  }

  std::string formatSummaryCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_binary_config: dtype=" << getDTypeID()
           << ", family=" << getFamilyID()
           << ", operator=" << getArithmeticVerb()
           << ", shape=" << getShapeID() << ", sew=" << getSEWBits()
           << ", lmul=" << getLMUL()
           << ", tail_policy=" << getTailPolicy()
           << ", mask_policy=" << getMaskPolicy()
           << ", vector_type=" << getVectorType()
           << ", vector_suffix=" << getVectorSuffix()
           << ", setvl_suffix=" << getSetVLSuffix()
           << ", runtime_element_count_c_name="
           << getRuntimeElementCountCName()
           << ", dispatch_availability_c_name="
           << getDispatchAvailabilityGuardCName();
    if (getComponentCapacityElementCount() > 0)
      stream << ", component_capacity_element_count=" << getComponentCapacityElementCount();
    if (fixedSourceExtent)
      stream << ", fixed_source_vector_extent="
             << fixedSourceExtent->sourceVectorExtent
             << ", runtime_element_count_constraint="
             << fixedSourceExtent->runtimeElementCountConstraint;
    if (dynamicRuntimeExtent)
      stream << ", runtime_extent_arg="
             << dynamicRuntimeExtent->runtimeExtentArg
             << ", source_loop_step="
             << dynamicRuntimeExtent->sourceLoopStep
             << ", source_vector_chunk_extent="
             << dynamicRuntimeExtent->sourceVectorChunkExtent
             << ", active_lane_authority="
             << dynamicRuntimeExtent->activeLaneAuthority
             << ", source_tail_policy="
             << dynamicRuntimeExtent->sourceTailPolicy
             << ", runtime_element_count_constraint="
             << dynamicRuntimeExtent->runtimeElementCountConstraint;
    if (!selectedVariantSymbol.empty())
      stream << ", selected_variant=@" << selectedVariantSymbol;
    if (!selectedRole.empty())
      stream << ", selected_role=" << selectedRole;
    stream.flush();
    return text;
  }

  std::string formatRuntimeVLBoundaryCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << runtimeLength.formatRuntimeVLBoundaryCommentBody();
    if (fixedSourceExtent)
      stream << ", fixed_source_vector_extent="
             << fixedSourceExtent->sourceVectorExtent
             << ", runtime_element_count_constraint="
             << fixedSourceExtent->runtimeElementCountConstraint;
    if (dynamicRuntimeExtent)
      stream << ", runtime_extent_arg="
             << dynamicRuntimeExtent->runtimeExtentArg
             << ", source_loop_step="
             << dynamicRuntimeExtent->sourceLoopStep
             << ", source_vector_chunk_extent="
             << dynamicRuntimeExtent->sourceVectorChunkExtent
             << ", active_lane_authority="
             << dynamicRuntimeExtent->activeLaneAuthority
             << ", source_tail_policy="
             << dynamicRuntimeExtent->sourceTailPolicy
             << ", runtime_element_count_constraint="
             << dynamicRuntimeExtent->runtimeElementCountConstraint;
    if (!selectedVariantSymbol.empty())
      stream << ", selected_variant=@" << selectedVariantSymbol;
    if (!selectedRole.empty())
      stream << ", selected_role=" << selectedRole;
    stream.flush();
    return text;
  }

  std::string formatDispatchContractSelectedVectorConfigMetadataValue() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "shape=" << getShapeID() << ",sew=" << getSEWBits()
           << ",lmul=" << getLMUL()
           << ",tail_policy=" << getTailPolicy()
           << ",mask_policy=" << getMaskPolicy()
           << ",vector_type=" << getVectorType()
           << ",vector_suffix=" << getVectorSuffix()
           << ",setvl_suffix=" << getSetVLSuffix();
    stream.flush();
    return text;
  }

  std::string formatDispatchContractRuntimeVLBoundaryMetadataValue() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "runtime_element_count_c_name="
           << getRuntimeElementCountCName()
           << ",runtime_avl_source=" << getRuntimeAVLSource()
           << ",runtime_avl_role=" << getRuntimeAVLRole()
           << ",runtime_vl_source=" << getRuntimeVLSource()
           << ",runtime_vl_scope=" << getRuntimeVLScope();
    stream.flush();
    return text;
  }

  std::string formatDispatchContractSelectedSourceIdentityMetadataValue(
      llvm::StringRef sourceKind, llvm::StringRef microkernelOp) const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "source_kind=" << sourceKind
           << ",dtype=" << getDTypeID()
           << ",family=" << getFamilyID()
           << ",operator=" << getArithmeticVerb()
           << ",microkernel_op=" << microkernelOp
           << ",emitc_source_op=" << getArithmeticOpName()
           << ",emitc_lowerable_op_interface=TCRVEmitCLowerableOpInterface";
    stream.flush();
    return text;
  }

private:
  const RVVBinaryFamilyRecord *family = nullptr;
  const RVVVectorShapeConfig *shape = nullptr;
  std::string selectedVariantSymbol;
  std::string selectedRole;
  RVVRuntimeLengthContract runtimeLength;
  std::string dispatchAvailabilityGuardCName = "rvv_available";
  std::optional<support::FixedVectorSourceExtentContract> fixedSourceExtent;
  std::optional<support::DynamicVectorRuntimeExtentContract>
      dynamicRuntimeExtent;
};

struct RVVBinarySelectedConfigProfile {
  const RVVBinarySelectedConfigContract *contract = nullptr;

  bool isValid() const { return contract && contract->isValid(); }

  std::string formatHardwareFactsMetadataValue() const {
    return contract
               ? contract->formatSelectedConfigProfileHardwareFactsMetadataValue()
               : std::string();
  }

  std::string formatVariantConfigMetadataValue() const {
    return contract
               ? contract->formatSelectedConfigProfileVariantConfigMetadataValue()
               : std::string();
  }

  std::string formatRuntimeRolesMetadataValue() const {
    return contract
               ? contract->formatSelectedConfigProfileRuntimeRolesMetadataValue()
               : std::string();
  }
};

struct RVVBinarySelectedConfigEmissionView {
  std::int64_t sew = 0;
  std::string lmul;
  std::string vectorType;
  std::string vectorSuffix;
  std::string setvlSuffix;
  std::string tailPolicy;
  std::string maskPolicy;

  bool isValid() const {
    return sew > 0 && !lmul.empty() && !vectorType.empty() &&
           !vectorSuffix.empty() && !setvlSuffix.empty() &&
           !tailPolicy.empty() && !maskPolicy.empty();
  }
};

struct RVVBinaryEmitCBodyMapping {
  std::string routeKind;
  std::string sourceAuthority;
  std::string requiredHeader;

  bool isValid() const {
    return !routeKind.empty() && !sourceAuthority.empty() &&
           !requiredHeader.empty();
  }
};

inline llvm::Error makeRVVSelectedConfigContractError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV selected config contract failed: ") +
          message,
      llvm::errc::invalid_argument);
}

inline llvm::Error validateRVVBinarySelectedConfigContract(
    const RVVBinarySelectedConfigContract &contract) {
  if (!contract.isValid())
    return makeRVVSelectedConfigContractError(
        "requires a finite binary family descriptor and selected vector-shape "
        "config");

  const RVVBinaryFamilyRecord &family = contract.getFamily();
  const RVVVectorShapeConfig &shape = contract.getShape();
  if (family.dtypeID != shape.dtypeID)
    return makeRVVSelectedConfigContractError(
        llvm::Twine("selected config mismatch: finite binary family '") +
        family.familyID + "' has dtype '" + family.dtypeID +
        "' but selected vector-shape '" + shape.shapeID + "' has dtype '" +
        shape.dtypeID + "'");

  const RVVVectorShapeConfig *registeredShape =
      lookupRVVBinaryFamilyShapeConfigByID(family, shape.shapeID);
  if (!registeredShape)
    return makeRVVSelectedConfigContractError(
        llvm::Twine("selected config mismatch: vector-shape '") +
        shape.shapeID + "' does not belong to finite binary family '" +
        family.familyID + "'");

  if (registeredShape->sewBits != shape.sewBits ||
      registeredShape->lmul != shape.lmul ||
      registeredShape->tailPolicy != shape.tailPolicy ||
      registeredShape->maskPolicy != shape.maskPolicy ||
      registeredShape->vectorType != shape.vectorType ||
      registeredShape->vectorSuffix != shape.vectorSuffix ||
      registeredShape->setvlSuffix != shape.setvlSuffix)
    return makeRVVSelectedConfigContractError(
        llvm::Twine("selected config mismatch: vector-shape '") +
        shape.shapeID +
        "' does not match the registered finite RVV vector config");

  if (!family.frontendContract)
    return makeRVVSelectedConfigContractError(
        llvm::Twine("selected config mismatch: finite binary family '") +
        family.familyID + "' is missing its frontend marker/ABI contract");

  if (family.frontendContract->dtypeID != family.dtypeID ||
      family.frontendContract->familyID != family.familyID ||
      family.frontendContract->frontendLowering != family.frontendLowering ||
      family.frontendContract->elementBitWidth != family.elementBitWidth ||
      family.frontendContract->constInputPointerCType !=
          family.constInputPointerCType ||
      family.frontendContract->outputPointerCType !=
          family.outputPointerCType)
    return makeRVVSelectedConfigContractError(
        llvm::Twine("selected config mismatch: finite binary family '") +
        family.familyID +
        "' disagrees with the source-derived frontend marker/ABI "
        "contract");

  if (llvm::Error error =
          validateRVVRuntimeLengthContract(contract.getRuntimeLengthContract())) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVSelectedConfigContractError(message);
  }

  if (contract.getComponentCapacityElementCount() < 0 ||
      contract.getComponentCapacityElementCount() > 64)
    return makeRVVSelectedConfigContractError(
        llvm::Twine("selected config mismatch: finite binary family '") +
        family.familyID +
        "' artifact-local component capacity must be in [1, 64] when present");

  if (contract.getRuntimeElementCountCName().empty())
    return makeRVVSelectedConfigContractError(
        "runtime element-count C name must be non-empty");
  if (contract.getDispatchAvailabilityGuardCName().empty())
    return makeRVVSelectedConfigContractError(
        "dispatch availability guard C name must be non-empty");
  if (contract.getFixedVectorSourceExtentContract()) {
    const support::FixedVectorSourceExtentContract &sourceExtent =
        *contract.getFixedVectorSourceExtentContract();
    if (!sourceExtent.isValid())
      return makeRVVSelectedConfigContractError(
          "fixed vector source extent contract is incomplete");
    if (contract.getComponentCapacityElementCount() <= 0)
      return makeRVVSelectedConfigContractError(
          "fixed vector source extent contract requires a positive "
          "artifact-local component capacity for cross-checking");
    if (sourceExtent.sourceVectorExtent != contract.getComponentCapacityElementCount())
      return makeRVVSelectedConfigContractError(
          llvm::Twine("fixed vector source extent ") +
          llvm::Twine(sourceExtent.sourceVectorExtent) +
          " must match artifact-local component capacity " +
          llvm::Twine(contract.getComponentCapacityElementCount()));
  }
  if (contract.getDynamicVectorRuntimeExtentContract()) {
    const support::DynamicVectorRuntimeExtentContract &runtimeExtent =
        *contract.getDynamicVectorRuntimeExtentContract();
    if (!runtimeExtent.isValid())
      return makeRVVSelectedConfigContractError(
          "dynamic vector runtime extent contract is incomplete");
    if (contract.getRuntimeElementCountCName() !=
        runtimeExtent.runtimeExtentArg)
      return makeRVVSelectedConfigContractError(
          llvm::Twine("dynamic vector runtime extent arg '") +
          runtimeExtent.runtimeExtentArg +
          "' must match runtime element-count C name '" +
          contract.getRuntimeElementCountCName() + "'");
  }
  if (contract.getFixedVectorSourceExtentContract() &&
      contract.getDynamicVectorRuntimeExtentContract())
    return makeRVVSelectedConfigContractError(
        "fixed source extent and dynamic runtime extent contracts are "
        "mutually exclusive");

  return llvm::Error::success();
}

inline llvm::Expected<RVVBinarySelectedConfigEmissionView>
buildRVVBinarySelectedConfigEmissionView(
    const RVVBinarySelectedConfigContract &contract) {
  if (llvm::Error error = validateRVVBinarySelectedConfigContract(contract))
    return std::move(error);

  RVVBinarySelectedConfigEmissionView view;
  view.sew = contract.getSEWBits();
  view.lmul = contract.getLMUL().str();
  view.vectorType = contract.getVectorType().str();
  view.vectorSuffix = contract.getVectorSuffix().str();
  view.setvlSuffix = contract.getSetVLSuffix().str();
  view.tailPolicy = contract.getTailPolicy().str();
  view.maskPolicy = contract.getMaskPolicy().str();
  if (!view.isValid())
    return makeRVVSelectedConfigContractError(
        "selected config emission view must contain non-empty vector type, "
        "suffix, and policy fields");
  return view;
}

inline llvm::Expected<RVVBinaryEmitCBodyMapping>
buildRVVBinaryEmitCBodyMappingFromSelectedConfig(
    const RVVBinarySelectedConfigContract &contract) {
  if (llvm::Error error = validateRVVBinarySelectedConfigContract(contract))
    return std::move(error);

  RVVBinaryEmitCBodyMapping mapping;
  mapping.routeKind = getRVVEmitCRouteKindMetadataValue().str();
  mapping.sourceAuthority = getRVVEmitCSourceAuthorityMetadataValue().str();
  mapping.requiredHeader = getRVVEmitCRequiredHeaderMetadataValue().str();
  if (!mapping.isValid())
    return makeRVVSelectedConfigContractError(
        "EmitC body mapping requires route kind, source authority, required "
        "header");
  return mapping;
}

inline llvm::Expected<RVVBinarySelectedConfigContract>
buildRVVBinarySelectedConfigContract(
    const RVVBinaryFamilyRecord &family,
    const RVVVectorShapeConfig &shape,
    llvm::StringRef selectedVariantSymbol = llvm::StringRef(),
    llvm::StringRef selectedRole = llvm::StringRef(),
    std::int64_t componentCapacityElementCount = 0,
    llvm::StringRef runtimeElementCountCName = "n",
    llvm::StringRef dispatchAvailabilityGuardCName = "rvv_available",
    std::optional<support::FixedVectorSourceExtentContract>
        fixedSourceExtent = std::nullopt,
    std::optional<support::DynamicVectorRuntimeExtentContract>
        dynamicRuntimeExtent = std::nullopt) {
  RVVBinarySelectedConfigContract contract(family, shape);
  contract.setSelectedPath(selectedVariantSymbol, selectedRole);
  contract.setComponentCapacityElementCount(componentCapacityElementCount);
  contract.setRuntimeElementCountCName(runtimeElementCountCName);
  contract.setDispatchAvailabilityGuardCName(dispatchAvailabilityGuardCName);
  contract.setFixedVectorSourceExtentContract(std::move(fixedSourceExtent));
  contract.setDynamicVectorRuntimeExtentContract(
      std::move(dynamicRuntimeExtent));
  if (llvm::Error error = validateRVVBinarySelectedConfigContract(contract))
    return std::move(error);
  return contract;
}

inline void appendRVVBinarySelectedVectorShapeMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  appendRVVVectorShapeSelectedPlanMetadata(contract.getShape(), out);
}

inline void appendRVVBinaryRuntimeVLBoundarySelectedPlanMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  appendRVVRuntimeLengthSelectedPlanMetadata(contract.getRuntimeLengthContract(),
                                            out);
}

inline void appendRVVBinaryFixedVectorSourceExtentSelectedPlanMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  if (!contract.getFixedVectorSourceExtentContract())
    return;

  const support::FixedVectorSourceExtentContract &sourceExtent =
      *contract.getFixedVectorSourceExtentContract();
  llvm::StringRef role = support::getFrontendSourceExtentMetadataRole();
  llvm::StringRef note = support::getFrontendSourceExtentMetadataNote();
  out.push_back({support::getFrontendSourceKindMetadataName(),
                 sourceExtent.sourceKind, role, note,
                 "frontend source kind"});
  out.push_back({support::getFrontendSourceAuthorityMetadataName(),
                 sourceExtent.sourceAuthority, role, note,
                 "frontend source authority"});
  out.push_back({support::getFrontendSourceVectorExtentMetadataName(),
                 std::to_string(sourceExtent.sourceVectorExtent), role, note,
                 "frontend source vector extent"});
  out.push_back(
      {support::getFrontendRuntimeElementCountConstraintMetadataName(),
       sourceExtent.runtimeElementCountConstraint, role, note,
       "frontend runtime element-count constraint"});
}

inline void appendRVVBinaryDynamicRuntimeExtentSelectedPlanMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  if (!contract.getDynamicVectorRuntimeExtentContract())
    return;

  const support::DynamicVectorRuntimeExtentContract &runtimeExtent =
      *contract.getDynamicVectorRuntimeExtentContract();
  llvm::StringRef role = support::getFrontendRuntimeExtentMetadataRole();
  llvm::StringRef note = support::getFrontendRuntimeExtentMetadataNote();
  out.push_back({support::getFrontendSourceKindMetadataName(),
                 runtimeExtent.sourceKind, role, note,
                 "frontend source kind"});
  out.push_back({support::getFrontendSourceAuthorityMetadataName(),
                 runtimeExtent.sourceAuthority, role, note,
                 "frontend source authority"});
  out.push_back({support::getFrontendRuntimeExtentArgMetadataName(),
                 runtimeExtent.runtimeExtentArg, role, note,
                 "frontend runtime extent arg"});
  out.push_back({support::getFrontendSourceLoopStepMetadataName(),
                 std::to_string(runtimeExtent.sourceLoopStep), role, note,
                 "frontend source loop step"});
  out.push_back({support::getFrontendSourceVectorChunkExtentMetadataName(),
                 std::to_string(runtimeExtent.sourceVectorChunkExtent), role,
                 note, "frontend source vector chunk extent"});
  out.push_back({support::getFrontendActiveLaneAuthorityMetadataName(),
                 runtimeExtent.activeLaneAuthority, role, note,
                 "frontend active-lane authority"});
  out.push_back({support::getFrontendSourceTailPolicyMetadataName(),
                 runtimeExtent.sourceTailPolicy, role, note,
                 "frontend source tail policy"});
  out.push_back(
      {support::getFrontendRuntimeElementCountConstraintMetadataName(),
       runtimeExtent.runtimeElementCountConstraint, role, note,
       "frontend runtime element-count constraint"});
}

inline void appendRVVBinarySelectedTypedSourceMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  llvm::StringRef typedRole = getRVVTypedBinarySourceMetadataRole();
  llvm::StringRef typedNote = getRVVTypedBinarySourceMetadataNote();
  out.push_back({getRVVSelectedBinaryDTypeMetadataName(),
                 contract.getDTypeID(), typedRole, typedNote,
                 "selected binary dtype"});
  out.push_back({getRVVSelectedBinaryFamilyMetadataName(),
                 contract.getFamilyID(), typedRole, typedNote,
                 "selected binary family"});
  out.push_back({getRVVSelectedBinaryOperatorMetadataName(),
                 contract.getArithmeticVerb(), typedRole, typedNote,
                 "selected binary operator"});
  out.push_back({getRVVEmitCSourceOpMetadataName(),
                 contract.getArithmeticOpName(),
                 getRVVEmitCSourceOpMetadataRole(),
                 getRVVEmitCSourceOpMetadataNote(),
                 "EmitC source op"});
  out.push_back({getRVVEmitCLowerableOpInterfaceMetadataName(),
                 "TCRVEmitCLowerableOpInterface",
                 getRVVEmitCSourceOpMetadataRole(),
                 getRVVEmitCLowerableOpInterfaceMetadataNote(),
                 "EmitC lowerable op interface"});
  out.push_back({getRVVRuntimeElementCountCNameMetadataName(),
                 contract.getRuntimeElementCountCName(),
                 getRVVRuntimeControlNameMetadataRole(),
                 getRVVRuntimeControlNameMetadataNote(),
                 "runtime element-count C name"});
}

inline void appendRVVBinaryEmitCRouteMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  llvm::StringRef routeRole = getRVVEmitCRouteMetadataRole();
  llvm::StringRef routeNote = getRVVEmitCRouteMetadataNote();
  out.push_back({getRVVEmitCRouteKindMetadataName(),
                 getRVVEmitCRouteKindMetadataValue(), routeRole, routeNote,
                 "EmitC route kind"});
  out.push_back({getRVVEmitCSourceAuthorityMetadataName(),
                 getRVVEmitCSourceAuthorityMetadataValue(), routeRole,
                 routeNote, "EmitC source authority"});
  out.push_back({getRVVEmitCRequiredHeaderMetadataName(),
                 getRVVEmitCRequiredHeaderMetadataValue(), routeRole,
                 routeNote, "EmitC required header"});
}

inline void appendRVVBinarySelectedConfigProfileMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  RVVBinarySelectedConfigProfile profile{&contract};
  llvm::StringRef role = getRVVSelectedConfigProfileMetadataRole();
  llvm::StringRef note = getRVVSelectedConfigProfileMetadataNote();
  out.push_back({getRVVSelectedConfigProfileHardwareFactsMetadataName(),
                 profile.formatHardwareFactsMetadataValue(), role, note,
                 "selected config profile hardware facts"});
  out.push_back({getRVVSelectedConfigProfileVariantConfigMetadataName(),
                 profile.formatVariantConfigMetadataValue(), role, note,
                 "selected config profile variant config"});
  out.push_back({getRVVSelectedConfigProfileRuntimeRolesMetadataName(),
                 profile.formatRuntimeRolesMetadataValue(), role, note,
                 "selected config profile runtime roles"});
}

inline void appendRVVBinarySelectedSourceIdentityMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::StringRef sourceKind,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  llvm::StringRef trimmedSourceKind = sourceKind.trim();
  if (trimmedSourceKind.empty())
    return;

  llvm::StringRef role = getRVVTypedBinarySourceIdentityMetadataRole();
  llvm::StringRef note = getRVVTypedBinarySourceIdentityMetadataNote();
  out.push_back({getRVVSelectedBinarySourceKindMetadataName(),
                 trimmedSourceKind, role, note,
                 "selected binary source kind"});
  out.push_back({getRVVSelectedBinaryMicrokernelOpMetadataName(),
                 contract.getFamily().microkernelOpName, role, note,
                 "selected binary microkernel op"});
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVSELECTEDCONFIGCONTRACT_H
