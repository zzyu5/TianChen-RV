#ifndef TIANCHENRV_TARGET_RVV_RVVSELECTEDCONFIGCONTRACT_H
#define TIANCHENRV_TARGET_RVV_RVVSELECTEDCONFIGCONTRACT_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

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

inline llvm::StringRef getRVVSelectedLoweringDescriptorMetadataName() {
  return "tcrv_rvv.selected_lowering_descriptor";
}

inline llvm::StringRef getRVVDescriptorElementCountMetadataName() {
  return "tcrv_rvv.descriptor_element_count";
}

inline llvm::StringRef getRVVRuntimeElementCountCNameMetadataName() {
  return "tcrv_rvv.runtime_element_count_c_name";
}

inline llvm::StringRef getRVVEmitCSourceOpMetadataName() {
  return "tcrv_rvv.emitc_source_op";
}

inline llvm::StringRef getRVVEmitCLowerableOpInterfaceMetadataName() {
  return "tcrv_rvv.emitc_lowerable_op_interface";
}

inline llvm::StringRef getRVVLegacyDescriptorMirrorMetadataRole() {
  return "legacy-rvv-binary-descriptor-mirror";
}

inline llvm::StringRef getRVVTypedBinarySourceMetadataRole() {
  return "typed-rvv-binary-source";
}

inline llvm::StringRef getRVVLegacyDescriptorMirrorMetadataNote() {
  return "legacy finite RVV descriptor mirror metadata checked after typed "
         "selected-plan authority is established; not compute, ABI, source, "
         "runtime AVL/VL, hardware capacity, or performance authority";
}

inline llvm::StringRef getRVVTypedBinarySourceMetadataNote() {
  return "typed RVV family-op metadata selected by the RVV plugin for the "
         "common EmitC route; not descriptor-owned computation, runtime "
         "correctness evidence, or performance evidence";
}

inline llvm::StringRef getRVVEmitCSourceOpMetadataRole() {
  return "typed-rvv-emitc-source-op";
}

inline llvm::StringRef getRVVEmitCSourceOpMetadataNote() {
  return "typed RVV arithmetic op used as the source operation for the common "
         "EmitC lowerable route; not a lowering descriptor";
}

inline llvm::StringRef getRVVEmitCLowerableOpInterfaceMetadataNote() {
  return "generated RVV op interface queried before building the common EmitC "
         "lowerable route; not descriptor-selected computation";
}

inline llvm::StringRef getRVVRuntimeControlNameMetadataRole() {
  return "rvv-runtime-control-name-boundary";
}

inline llvm::StringRef getRVVRuntimeControlNameMetadataNote() {
  return "runtime ABI/control C name resolved from tcrv.exec runtime boundary; "
         "not a compile-time vector-shape config or descriptor element_count";
}

class RVVBinarySelectedConfigContract {
public:
  RVVBinarySelectedConfigContract() = default;

  RVVBinarySelectedConfigContract(const RVVBinaryFamilyDescriptor &family,
                                  const RVVVectorShapeConfig &shape)
      : family(&family), shape(&shape) {}

  bool isValid() const { return family && shape; }

  const RVVBinaryFamilyDescriptor &getFamily() const { return *family; }
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
  llvm::StringRef getLegacyLoweringDescriptorMirror() const {
    return family ? family->loweringDescriptor : llvm::StringRef();
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
  std::int64_t getDescriptorElementCount() const {
    return descriptorElementCount;
  }
  llvm::StringRef getRuntimeElementCountCName() const {
    return runtimeElementCountCName;
  }
  llvm::StringRef getDispatchAvailabilityGuardCName() const {
    return dispatchAvailabilityGuardCName;
  }

  void setSelectedPath(llvm::StringRef variantSymbol, llvm::StringRef role) {
    selectedVariantSymbol = variantSymbol.trim().str();
    selectedRole = role.trim().str();
  }

  void setDescriptorElementCount(std::int64_t count) {
    descriptorElementCount = count;
  }

  void setRuntimeElementCountCName(llvm::StringRef cName) {
    llvm::StringRef trimmed = cName.trim();
    runtimeElementCountCName = trimmed.empty() ? "n" : trimmed.str();
  }

  void setDispatchAvailabilityGuardCName(llvm::StringRef cName) {
    llvm::StringRef trimmed = cName.trim();
    dispatchAvailabilityGuardCName =
        trimmed.empty() ? "rvv_available" : trimmed.str();
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

  llvm::SmallVector<support::RuntimeABIParameter, 4>
  getCallableRuntimeABIParameters() const {
    return getRVVBinaryCallableRuntimeABIParameters(
        getFamily(), runtimeElementCountCName);
  }

  llvm::SmallVector<support::RuntimeABIParamSpec, 1>
  getRuntimeElementCountParamSpecs() const {
    return getRVVBinaryRuntimeElementCountParamSpecs(
        getFamily(), runtimeElementCountCName);
  }

  support::RuntimeABIParameter getDispatchAvailabilityGuardParameter() const {
    return support::makeTargetExportABIParameter(
        dispatchAvailabilityGuardCName, "int",
        support::RuntimeABIParameterRole::DispatchAvailabilityGuard);
  }

  RVVBinaryIntrinsicDescriptor getIntrinsicDescriptor() const {
    return getRVVBinaryIntrinsicDescriptor(getFamily(), getShape());
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
    if (descriptorElementCount > 0)
      stream << ", descriptor_element_count=" << descriptorElementCount;
    if (!selectedVariantSymbol.empty())
      stream << ", selected_variant=@" << selectedVariantSymbol;
    if (!selectedRole.empty())
      stream << ", selected_role=" << selectedRole;
    stream.flush();
    return text;
  }

private:
  const RVVBinaryFamilyDescriptor *family = nullptr;
  const RVVVectorShapeConfig *shape = nullptr;
  std::string selectedVariantSymbol;
  std::string selectedRole;
  std::int64_t descriptorElementCount = 0;
  std::string runtimeElementCountCName = "n";
  std::string dispatchAvailabilityGuardCName = "rvv_available";
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

  const RVVBinaryFamilyDescriptor &family = contract.getFamily();
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

  if (contract.getDescriptorElementCount() < 0 ||
      contract.getDescriptorElementCount() > 64)
    return makeRVVSelectedConfigContractError(
        llvm::Twine("selected config mismatch: finite binary family '") +
        family.familyID +
        "' descriptor-local element_count must be in [1, 64] when present");

  if (contract.getRuntimeElementCountCName().empty())
    return makeRVVSelectedConfigContractError(
        "runtime element-count C name must be non-empty");
  if (contract.getDispatchAvailabilityGuardCName().empty())
    return makeRVVSelectedConfigContractError(
        "dispatch availability guard C name must be non-empty");

  return llvm::Error::success();
}

inline llvm::Expected<RVVBinarySelectedConfigContract>
buildRVVBinarySelectedConfigContract(
    const RVVBinaryFamilyDescriptor &family,
    const RVVVectorShapeConfig &shape,
    llvm::StringRef selectedVariantSymbol = llvm::StringRef(),
    llvm::StringRef selectedRole = llvm::StringRef(),
    std::int64_t descriptorElementCount = 0,
    llvm::StringRef runtimeElementCountCName = "n",
    llvm::StringRef dispatchAvailabilityGuardCName = "rvv_available") {
  RVVBinarySelectedConfigContract contract(family, shape);
  contract.setSelectedPath(selectedVariantSymbol, selectedRole);
  contract.setDescriptorElementCount(descriptorElementCount);
  contract.setRuntimeElementCountCName(runtimeElementCountCName);
  contract.setDispatchAvailabilityGuardCName(dispatchAvailabilityGuardCName);
  if (llvm::Error error = validateRVVBinarySelectedConfigContract(contract))
    return std::move(error);
  return contract;
}

inline void appendRVVBinaryLegacyDescriptorMirrorMetadata(
    const RVVBinarySelectedConfigContract &contract,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  llvm::StringRef descriptorRole =
      getRVVLegacyDescriptorMirrorMetadataRole();
  llvm::StringRef descriptorNote =
      getRVVLegacyDescriptorMirrorMetadataNote();
  out.push_back({getRVVSelectedBinaryDTypeMetadataName(),
                 contract.getDTypeID(), descriptorRole, descriptorNote,
                 "selected binary dtype"});
  out.push_back({getRVVSelectedBinaryFamilyMetadataName(),
                 contract.getFamilyID(), descriptorRole, descriptorNote,
                 "selected binary family"});
  out.push_back({getRVVSelectedBinaryOperatorMetadataName(),
                 contract.getArithmeticVerb(), descriptorRole, descriptorNote,
                 "selected binary operator"});
  out.push_back({getRVVSelectedLoweringDescriptorMetadataName(),
                 contract.getLegacyLoweringDescriptorMirror(), descriptorRole,
                 descriptorNote, "legacy lowering descriptor mirror"});
  out.push_back({getRVVRuntimeElementCountCNameMetadataName(),
                 contract.getRuntimeElementCountCName(),
                 getRVVRuntimeControlNameMetadataRole(),
                 getRVVRuntimeControlNameMetadataNote(),
                 "runtime element-count C name"});
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

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVSELECTEDCONFIGCONTRACT_H
