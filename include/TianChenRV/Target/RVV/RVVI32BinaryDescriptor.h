#ifndef TIANCHENRV_TARGET_RVV_RVVI32BINARYDESCRIPTOR_H
#define TIANCHENRV_TARGET_RVV_RVVI32BINARYDESCRIPTOR_H

#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace tianchenrv::target::rvv {

struct RVVI32BinaryIntrinsicDescriptor {
  const i32_binary::I32BinaryFamilyDescriptor *family = nullptr;
  const RVVI32VectorShapeConfig *shape = nullptr;

  const i32_binary::RVVI32MicrokernelFamilyDescriptor &
  getRVVFamily() const {
    return family->rvv;
  }

  const i32_binary::DispatchI32FamilyDescriptor &
  getDispatchFamily() const {
    return family->dispatch;
  }

  llvm::StringRef getArithmeticFamilyID() const { return family->familyID; }
  llvm::StringRef getLegacyLoweringDescriptorMirror() const {
    return family->loweringDescriptor;
  }
  llvm::StringRef getRVVOperationName() const {
    return family->rvv.arithmeticOpName;
  }
  llvm::StringRef getRVVMicrokernelOpName() const {
    return family->rvv.microkernelOpName;
  }
  llvm::StringRef getRVVRouteID() const { return family->rvv.routeID; }
  llvm::StringRef getRVVHeaderRouteID() const {
    return family->rvv.headerRouteID;
  }
  llvm::StringRef getRVVObjectRouteID() const {
    return family->rvv.objectRouteID;
  }
  llvm::StringRef getRVVRuntimeABIName() const {
    return family->rvv.runtimeABIName;
  }
  llvm::StringRef getRVVRuntimeGlueRole() const {
    return family->rvv.runtimeGlueRole;
  }
  llvm::StringRef getScalarRouteID() const {
    return family->scalar.routeID;
  }
  llvm::StringRef getScalarRuntimeABIName() const {
    return family->scalar.runtimeABIName;
  }
  llvm::StringRef getDispatchSourceRouteID() const {
    return family->dispatch.dispatchSourceRouteID;
  }
  llvm::StringRef getDispatchHeaderRouteID() const {
    return family->dispatch.dispatchHeaderRouteID;
  }
  llvm::StringRef getDispatchObjectRouteID() const {
    return family->dispatch.dispatchObjectRouteID;
  }
  llvm::StringRef getDispatchRuntimeABIName() const {
    return family->dispatch.dispatchRuntimeABIName;
  }
  llvm::StringRef getDispatchExternalABIComponentGroup() const {
    return family->dispatch.dispatchExternalABIComponentGroup;
  }
  llvm::StringRef getDispatchSuccessMarker() const {
    return family->dispatch.selfCheckSuccessMarker;
  }
  llvm::StringRef getVectorType() const { return shape->vectorType; }
  llvm::StringRef getVectorSuffix() const { return shape->vectorSuffix; }
  llvm::StringRef getSetVLSuffix() const { return shape->setvlSuffix; }
  llvm::StringRef getShapeID() const { return shape->shapeID; }
  std::int64_t getSEWBits() const { return shape->sewBits; }
  llvm::StringRef getLMUL() const { return shape->lmul; }
  llvm::StringRef getTailPolicy() const { return shape->tailPolicy; }
  llvm::StringRef getMaskPolicy() const { return shape->maskPolicy; }

  llvm::SmallVector<llvm::StringRef, 4> getSelectedShapeCapabilityIDs() const {
    llvm::SmallVector<llvm::StringRef, 4> capabilityIDs;
    capabilityIDs.push_back(shape->sewCapabilityID);
    capabilityIDs.push_back(shape->lmulCapabilityID);
    capabilityIDs.push_back(shape->tailPolicyCapabilityID);
    capabilityIDs.push_back(shape->maskPolicyCapabilityID);
    return capabilityIDs;
  }

  std::string getSetVLIntrinsicName() const {
    return (llvm::Twine("__riscv_vsetvl_") + shape->setvlSuffix).str();
  }

  std::string getLoadIntrinsicName() const {
    return (llvm::Twine("__riscv_vle32_v_") + shape->vectorSuffix).str();
  }

  std::string getArithmeticIntrinsicName() const {
    return (llvm::Twine(family->rvv.arithmeticIntrinsicPrefix) +
            shape->vectorSuffix)
        .str();
  }

  std::string getStoreIntrinsicName() const {
    return (llvm::Twine("__riscv_vse32_v_") + shape->vectorSuffix).str();
  }

  std::string formatSelectedVectorShapeConfigCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_vector_shape_config: shape=" << shape->shapeID
           << ", sew=" << shape->sewBits << ", lmul=" << shape->lmul
           << ", tail_policy=" << shape->tailPolicy
           << ", mask_policy=" << shape->maskPolicy
           << ", vector_type=" << shape->vectorType
           << ", vector_suffix=" << shape->vectorSuffix
           << ", setvl_suffix=" << shape->setvlSuffix;
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

  std::string formatIntrinsicConfigCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "intrinsic_config: vector_type=" << shape->vectorType
           << ", vector_suffix=" << shape->vectorSuffix
           << ", setvl_suffix=" << shape->setvlSuffix
           << ", tail_policy=" << shape->tailPolicy
           << ", mask_policy=" << shape->maskPolicy;
    stream.flush();
    return text;
  }
};

inline const i32_binary::I32BinaryFamilyDescriptor &
getI32BinaryFamilyDescriptorForRVV(
    const i32_binary::RVVI32MicrokernelFamilyDescriptor &family) {
  using Kind = i32_binary::I32BinaryFamilyKind;
  switch (family.kind) {
  case Kind::Add:
    return i32_binary::getI32VAddFamilyRegistrationRecord();
  case Kind::Sub:
    return i32_binary::getI32VSubFamilyRegistrationRecord();
  case Kind::Mul:
    return i32_binary::getI32VMulFamilyRegistrationRecord();
  }
  llvm_unreachable("unknown RVV i32 binary family");
}

inline const i32_binary::I32BinaryFamilyDescriptor &
getI32BinaryFamilyDescriptorForDispatch(
    const i32_binary::DispatchI32FamilyDescriptor &family) {
  using Kind = i32_binary::I32BinaryFamilyKind;
  switch (family.kind) {
  case Kind::Add:
    return i32_binary::getI32VAddFamilyRegistrationRecord();
  case Kind::Sub:
    return i32_binary::getI32VSubFamilyRegistrationRecord();
  case Kind::Mul:
    return i32_binary::getI32VMulFamilyRegistrationRecord();
  }
  llvm_unreachable("unknown RVV+scalar dispatch i32 binary family");
}

inline RVVI32BinaryIntrinsicDescriptor getRVVI32BinaryIntrinsicDescriptor(
    const i32_binary::I32BinaryFamilyDescriptor &family,
    const RVVI32VectorShapeConfig &shape) {
  return RVVI32BinaryIntrinsicDescriptor{&family, &shape};
}

inline RVVI32BinaryIntrinsicDescriptor getRVVI32BinaryIntrinsicDescriptor(
    const i32_binary::RVVI32MicrokernelFamilyDescriptor &family,
    const RVVI32VectorShapeConfig &shape) {
  return getRVVI32BinaryIntrinsicDescriptor(
      getI32BinaryFamilyDescriptorForRVV(family), shape);
}

inline RVVI32BinaryIntrinsicDescriptor getRVVI32BinaryIntrinsicDescriptor(
    const i32_binary::DispatchI32FamilyDescriptor &family,
    const RVVI32VectorShapeConfig &shape) {
  return getRVVI32BinaryIntrinsicDescriptor(
      getI32BinaryFamilyDescriptorForDispatch(family), shape);
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVI32BINARYDESCRIPTOR_H
