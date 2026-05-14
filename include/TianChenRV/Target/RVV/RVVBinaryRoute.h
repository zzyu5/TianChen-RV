#ifndef TIANCHENRV_TARGET_RVV_RVVBINARYROUTE_H
#define TIANCHENRV_TARGET_RVV_RVVBINARYROUTE_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamily.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace tianchenrv::target::rvv {

class RVVBinaryRuntimeABIContract
    : public support::FiniteBinaryRuntimeABIContract {
public:
  explicit RVVBinaryRuntimeABIContract(
      const RVVBinaryFamilyRecord &family)
      : support::FiniteBinaryRuntimeABIContract(
            support::FiniteBinaryRuntimeABIContractSpec{
                family.familyID,
                family.constInputPointerCType,
                family.outputPointerCType,
                {family.runtimeABI, family.runtimeABIKind,
                 family.runtimeABIName, family.runtimeGlueRole},
                {family.runtimeABI, family.runtimeABIKind,
                 family.runtimeABIName, family.runtimeGlueRole},
                {},
                {},
                family.externalABIComponentGroup}),
        family(&family) {}

  const RVVBinaryFamilyRecord &getFamilyRegistrationRecord() const {
    return *family;
  }

private:
  const RVVBinaryFamilyRecord *family = nullptr;
};

inline const RVVBinaryRuntimeABIContract &
getRVVBinaryRuntimeABIContract(const RVVBinaryFamilyRecord &family) {
  switch (family.dtype) {
  case RVVBinaryDTypeKind::I32:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add: {
      static const RVVBinaryRuntimeABIContract contract(
          getI32VAddFamilyRegistrationRecord());
      return contract;
    }
    case RVVBinaryArithmeticKind::Sub: {
      static const RVVBinaryRuntimeABIContract contract(
          getI32VSubFamilyRegistrationRecord());
      return contract;
    }
    case RVVBinaryArithmeticKind::Mul: {
      static const RVVBinaryRuntimeABIContract contract(
          getI32VMulFamilyRegistrationRecord());
      return contract;
    }
    }
    break;
  case RVVBinaryDTypeKind::I64:
    switch (family.arithmetic) {
    case RVVBinaryArithmeticKind::Add: {
      static const RVVBinaryRuntimeABIContract contract(
          getI64VAddFamilyRegistrationRecord());
      return contract;
    }
    case RVVBinaryArithmeticKind::Sub: {
      static const RVVBinaryRuntimeABIContract contract(
          getI64VSubFamilyRegistrationRecord());
      return contract;
    }
    case RVVBinaryArithmeticKind::Mul: {
      static const RVVBinaryRuntimeABIContract contract(
          getI64VMulFamilyRegistrationRecord());
      return contract;
    }
    }
    break;
  }
  llvm_unreachable("unknown RVV binary family descriptor");
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVBinaryCallableRuntimeABIParameters(
    const RVVBinaryFamilyRecord &family,
    llvm::StringRef runtimeCountCName = "n") {
  return getRVVBinaryRuntimeABIContract(family).getCallableParameters(
      runtimeCountCName);
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVBinaryCallableRuntimeABIRoleRequirements(
    const RVVBinaryFamilyRecord &family) {
  llvm::ArrayRef<support::RuntimeABIParameter> requirements =
      getRVVBinaryRuntimeABIContract(family).getCallableRoleRequirements();
  return llvm::SmallVector<support::RuntimeABIParameter, 4>(
      requirements.begin(), requirements.end());
}

inline llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3>
getRVVBinaryBufferMemWindowSpecs(
    const RVVBinaryFamilyRecord &family) {
  llvm::ArrayRef<support::RuntimeABIMemWindowSpec> specs =
      getRVVBinaryRuntimeABIContract(family).getBufferMemWindowSpecs();
  return llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3>(specs.begin(),
                                                                specs.end());
}

inline llvm::SmallVector<support::RuntimeABIParamSpec, 1>
getRVVBinaryRuntimeElementCountParamSpecs(
    const RVVBinaryFamilyRecord &family, llvm::StringRef cName = "n") {
  return getRVVBinaryRuntimeABIContract(family)
      .getRuntimeElementCountParamSpecs(cName);
}

struct RVVBinaryIntrinsicRoute {
  RVVBinaryFamilyRecord family;
  const RVVVectorShapeConfig *shape = nullptr;

  llvm::StringRef getDTypeID() const { return family.dtypeID; }
  llvm::StringRef getArithmeticFamilyID() const { return family.familyID; }
  llvm::StringRef getLegacyLoweringTokenMirror() const {
    return family.legacyLoweringToken;
  }
  llvm::StringRef getRVVOperationName() const {
    return family.arithmeticOpName;
  }
  llvm::StringRef getRVVMicrokernelOpName() const {
    return family.microkernelOpName;
  }
  llvm::StringRef getRVVRouteID() const { return family.routeID; }
  llvm::StringRef getRVVHeaderRouteID() const {
    return family.headerRouteID;
  }
  llvm::StringRef getRVVObjectRouteID() const {
    return family.objectRouteID;
  }
  llvm::StringRef getRVVRuntimeABI() const { return family.runtimeABI; }
  llvm::StringRef getRVVRuntimeABIKind() const {
    return family.runtimeABIKind;
  }
  llvm::StringRef getRVVRuntimeABIName() const {
    return family.runtimeABIName;
  }
  llvm::StringRef getRVVRuntimeGlueRole() const {
    return family.runtimeGlueRole;
  }
  llvm::StringRef getRVVExternalABIComponentGroup() const {
    return family.externalABIComponentGroup;
  }
  llvm::StringRef getConstInputPointerCType() const {
    return family.constInputPointerCType;
  }
  llvm::StringRef getOutputPointerCType() const {
    return family.outputPointerCType;
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
    return (llvm::Twine("__riscv_vle") + llvm::Twine(shape->sewBits) +
            "_v_" + shape->vectorSuffix)
        .str();
  }

  std::string getArithmeticIntrinsicName() const {
    return (llvm::Twine(family.arithmeticIntrinsicPrefix) +
            shape->vectorSuffix)
        .str();
  }

  std::string getStoreIntrinsicName() const {
    return (llvm::Twine("__riscv_vse") + llvm::Twine(shape->sewBits) +
            "_v_" + shape->vectorSuffix)
        .str();
  }

  llvm::SmallVector<support::RuntimeABIParameter, 4>
  getCallableRuntimeABIParameters(llvm::StringRef runtimeCountCName = "n")
      const {
    return getRVVBinaryCallableRuntimeABIParameters(family,
                                                    runtimeCountCName);
  }

  llvm::SmallVector<support::RuntimeABIParameter, 4>
  getCallableRuntimeABIRoleRequirements() const {
    return getRVVBinaryCallableRuntimeABIRoleRequirements(family);
  }

  llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3>
  getBufferMemWindowSpecs() const {
    return getRVVBinaryBufferMemWindowSpecs(family);
  }

  llvm::SmallVector<support::RuntimeABIParamSpec, 1>
  getRuntimeElementCountParamSpecs(llvm::StringRef cName = "n") const {
    return getRVVBinaryRuntimeElementCountParamSpecs(family, cName);
  }

  std::string formatSelectedVectorShapeConfigCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_vector_shape_config: ";
    if (family.dtype != RVVBinaryDTypeKind::I32)
      stream << "dtype=" << family.dtypeID << ", ";
    stream << "shape=" << shape->shapeID << ", sew=" << shape->sewBits
           << ", lmul=" << shape->lmul
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

inline RVVBinaryIntrinsicRoute getRVVBinaryIntrinsicRoute(
    const RVVBinaryFamilyRecord &family,
    const RVVVectorShapeConfig &shape) {
  return RVVBinaryIntrinsicRoute{family, &shape};
}

inline RVVBinaryIntrinsicRoute getI64VAddIntrinsicRoute() {
  return getRVVBinaryIntrinsicRoute(getI64VAddFamilyRegistrationRecord(),
                                         getI64M1VectorShapeConfig());
}

inline RVVBinaryIntrinsicRoute getI64VSubIntrinsicRoute() {
  return getRVVBinaryIntrinsicRoute(getI64VSubFamilyRegistrationRecord(),
                                         getI64M1VectorShapeConfig());
}

inline RVVBinaryIntrinsicRoute getI64VMulIntrinsicRoute() {
  return getRVVBinaryIntrinsicRoute(getI64VMulFamilyRegistrationRecord(),
                                         getI64M1VectorShapeConfig());
}

inline RVVBinaryIntrinsicRoute getI32VAddIntrinsicRoute(
    const RVVI32VectorShapeConfig &shape = getI32M1VectorShapeConfig()) {
  return getRVVBinaryIntrinsicRoute(getI32VAddFamilyRegistrationRecord(), shape);
}

inline RVVBinaryIntrinsicRoute getI32VSubIntrinsicRoute(
    const RVVI32VectorShapeConfig &shape = getI32M1VectorShapeConfig()) {
  return getRVVBinaryIntrinsicRoute(getI32VSubFamilyRegistrationRecord(), shape);
}

inline RVVBinaryIntrinsicRoute getI32VMulIntrinsicRoute(
    const RVVI32VectorShapeConfig &shape = getI32M1VectorShapeConfig()) {
  return getRVVBinaryIntrinsicRoute(getI32VMulFamilyRegistrationRecord(), shape);
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVBINARYROUTE_H
