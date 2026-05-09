#ifndef TIANCHENRV_TARGET_RVV_RVVBINARYDESCRIPTOR_H
#define TIANCHENRV_TARGET_RVV_RVVBINARYDESCRIPTOR_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
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

enum class RVVBinaryDTypeKind {
  I32,
  I64,
};

enum class RVVBinaryArithmeticKind {
  Add,
  Sub,
  Mul,
};

struct RVVBinaryFamilyDescriptor {
  RVVBinaryDTypeKind dtype = RVVBinaryDTypeKind::I32;
  RVVBinaryArithmeticKind arithmetic = RVVBinaryArithmeticKind::Add;
  llvm::StringRef dtypeID;
  llvm::StringRef familyID;
  llvm::StringRef frontendLowering;
  llvm::StringRef loweringDescriptor;
  llvm::StringRef descriptorNoun;
  llvm::StringRef microkernelOpName;
  llvm::StringRef arithmeticOpName;
  llvm::StringRef arithmeticVerb;
  llvm::StringRef functionStem;
  llvm::StringRef headerGuardStem;
  llvm::StringRef arithmeticIntrinsicPrefix;
  llvm::StringRef resultCName;
  llvm::StringRef emissionKind;
  llvm::StringRef routeID;
  llvm::StringRef headerRouteID;
  llvm::StringRef objectRouteID;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef externalABIComponentGroup;
  llvm::StringRef cOperator;
  llvm::StringRef scalarCType;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
};

inline RVVBinaryArithmeticKind
convertI32ArithmeticKind(i32_binary::I32BinaryFamilyKind kind) {
  switch (kind) {
  case i32_binary::I32BinaryFamilyKind::Add:
    return RVVBinaryArithmeticKind::Add;
  case i32_binary::I32BinaryFamilyKind::Sub:
    return RVVBinaryArithmeticKind::Sub;
  case i32_binary::I32BinaryFamilyKind::Mul:
    return RVVBinaryArithmeticKind::Mul;
  }
  llvm_unreachable("unknown i32 binary family kind");
}

inline RVVBinaryFamilyDescriptor
getRVVBinaryFamilyDescriptor(
    const i32_binary::I32BinaryFamilyDescriptor &family) {
  return RVVBinaryFamilyDescriptor{
      RVVBinaryDTypeKind::I32,
      convertI32ArithmeticKind(family.kind),
      "i32",
      family.familyID,
      family.frontendLowering,
      family.loweringDescriptor,
      family.descriptorNoun,
      family.rvv.microkernelOpName,
      family.rvv.arithmeticOpName,
      family.rvv.arithmeticVerb,
      family.rvv.functionStem,
      family.rvv.headerGuardStem,
      family.rvv.arithmeticIntrinsicPrefix,
      family.rvv.resultCName,
      family.rvv.emissionKind,
      family.rvv.routeID,
      family.rvv.headerRouteID,
      family.rvv.objectRouteID,
      family.rvv.runtimeABI,
      family.rvv.runtimeABIKind,
      family.rvv.runtimeABIName,
      family.rvv.runtimeGlueRole,
      family.rvv.externalABIComponentGroup,
      family.dispatch.cOperator,
      "int32_t",
      "const int32_t *",
      "int32_t *"};
}

inline const RVVBinaryFamilyDescriptor &getI64VAddFamilyDescriptor() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Add,
      "i64",
      "i64-vadd",
      "i64-vadd",
      "i64-vadd-microkernel.v1",
      "finite i64-vadd lowering descriptor",
      "tcrv_rvv.i64_vadd_microkernel",
      "tcrv_rvv.i64_add",
      "add",
      "i64_vadd",
      "I64_VADD",
      "__riscv_vadd_vv_",
      "sum_vec",
      "rvv-explicit-i64-vadd-microkernel-c-source",
      "tcrv-export-rvv-i64-vadd-microkernel-c",
      "tcrv-export-rvv-i64-vadd-microkernel-header",
      "tcrv-export-rvv-i64-vadd-microkernel-object",
      "rvv-i64-vadd-runtime-callable-c-abi.v1",
      "rvv-runtime-callable-c-abi",
      "rvv-i64-vadd-runtime-callable-c-function.v1",
      "runtime-callable-i64-vadd-function",
      "rvv-i64-vadd-microkernel-external-abi.v1",
      "+",
      "int64_t",
      "const int64_t *",
      "int64_t *"};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyByFrontendLowering(llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  if (frontendLowering == getI64VAddFamilyDescriptor().frontendLowering)
    return &getI64VAddFamilyDescriptor();
  return nullptr;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyByLoweringDescriptor(llvm::StringRef loweringDescriptor) {
  loweringDescriptor = loweringDescriptor.trim();
  if (loweringDescriptor == getI64VAddFamilyDescriptor().loweringDescriptor)
    return &getI64VAddFamilyDescriptor();
  return nullptr;
}

struct RVVBinaryIntrinsicDescriptor {
  RVVBinaryFamilyDescriptor family;
  const RVVVectorShapeConfig *shape = nullptr;

  llvm::StringRef getDTypeID() const { return family.dtypeID; }
  llvm::StringRef getArithmeticFamilyID() const { return family.familyID; }
  llvm::StringRef getLoweringDescriptor() const {
    return family.loweringDescriptor;
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
  llvm::StringRef getCOperator() const { return family.cOperator; }
  llvm::StringRef getScalarCType() const { return family.scalarCType; }
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
    llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
    parameters.push_back(support::makeTargetExportABIParameter(
        "lhs", family.constInputPointerCType,
        support::RuntimeABIParameterRole::LHSInputBuffer));
    parameters.push_back(support::makeTargetExportABIParameter(
        "rhs", family.constInputPointerCType,
        support::RuntimeABIParameterRole::RHSInputBuffer));
    parameters.push_back(support::makeTargetExportABIParameter(
        "out", family.outputPointerCType,
        support::RuntimeABIParameterRole::OutputBuffer));
    parameters.push_back(support::makeTargetExportABIParameter(
        runtimeCountCName, "size_t",
        support::RuntimeABIParameterRole::RuntimeElementCount));
    return parameters;
  }

  llvm::SmallVector<support::RuntimeABIParameter, 4>
  getCallableRuntimeABIRoleRequirements() const {
    llvm::SmallVector<support::RuntimeABIParameter, 4> requirements;
    for (const support::RuntimeABIParameter &parameter :
         getCallableRuntimeABIParameters())
      requirements.push_back(support::makeTargetExportABIRoleRequirement(
          parameter.cType, parameter.role));
    return requirements;
  }

  llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3>
  getBufferMemWindowSpecs() const {
    llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3> specs;
    specs.push_back(support::RuntimeABIMemWindowSpec(
        "abi_lhs_input_buffer",
        support::RuntimeABIParameterRole::LHSInputBuffer,
        support::kRuntimeABIReadAccess,
        support::stringifyRuntimeABIParameterOwnership(
            support::RuntimeABIParameterOwnership::TargetExportABIOwned),
        family.constInputPointerCType));
    specs.push_back(support::RuntimeABIMemWindowSpec(
        "abi_rhs_input_buffer",
        support::RuntimeABIParameterRole::RHSInputBuffer,
        support::kRuntimeABIReadAccess,
        support::stringifyRuntimeABIParameterOwnership(
            support::RuntimeABIParameterOwnership::TargetExportABIOwned),
        family.constInputPointerCType));
    specs.push_back(support::RuntimeABIMemWindowSpec(
        "abi_output_buffer",
        support::RuntimeABIParameterRole::OutputBuffer,
        support::kRuntimeABIWriteAccess,
        support::stringifyRuntimeABIParameterOwnership(
            support::RuntimeABIParameterOwnership::TargetExportABIOwned),
        family.outputPointerCType));
    return specs;
  }

  llvm::SmallVector<support::RuntimeABIParamSpec, 1>
  getRuntimeElementCountParamSpecs(llvm::StringRef cName = "n") const {
    llvm::SmallVector<support::RuntimeABIParamSpec, 1> specs;
    specs.push_back(support::RuntimeABIParamSpec(
        "abi_runtime_element_count",
        support::RuntimeABIParameterRole::RuntimeElementCount, cName, "size_t",
        support::stringifyRuntimeABIParameterOwnership(
            support::RuntimeABIParameterOwnership::TargetExportABIOwned)));
    return specs;
  }

  std::string getCArithmeticCheckExpression(llvm::StringRef lhs,
                                            llvm::StringRef rhs) const {
    return (llvm::Twine(lhs) + " " + family.cOperator + " " + rhs).str();
  }

  std::string formatSelectedVectorShapeConfigCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_vector_shape_config: dtype=" << family.dtypeID
           << ", shape=" << shape->shapeID << ", sew=" << shape->sewBits
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
    stream << "intrinsic_config: dtype=" << family.dtypeID
           << ", vector_type=" << shape->vectorType
           << ", vector_suffix=" << shape->vectorSuffix
           << ", setvl_suffix=" << shape->setvlSuffix
           << ", tail_policy=" << shape->tailPolicy
           << ", mask_policy=" << shape->maskPolicy;
    stream.flush();
    return text;
  }
};

inline RVVBinaryIntrinsicDescriptor getRVVBinaryIntrinsicDescriptor(
    const RVVBinaryFamilyDescriptor &family,
    const RVVVectorShapeConfig &shape) {
  return RVVBinaryIntrinsicDescriptor{family, &shape};
}

inline RVVBinaryIntrinsicDescriptor getI64VAddIntrinsicDescriptor() {
  return getRVVBinaryIntrinsicDescriptor(getI64VAddFamilyDescriptor(),
                                         getI64M1VectorShapeConfig());
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVBINARYDESCRIPTOR_H
