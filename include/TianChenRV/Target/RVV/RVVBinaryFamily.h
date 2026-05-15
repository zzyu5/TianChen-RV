#ifndef TIANCHENRV_TARGET_RVV_RVVBINARYFAMILY_H
#define TIANCHENRV_TARGET_RVV_RVVBINARYFAMILY_H

#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

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

// Finite RVV binary family facts used only after typed RVV extension-family
// body structure has already selected a bounded family. Frontend-lowering
// markers, selected-source metadata, route ids, intrinsic spellings, artifact
// kinds, runtime ABI identity strings, and direct exporter names are
// intentionally not part of this record.
struct RVVBinaryFamilyRecord {
  RVVBinaryDTypeKind dtype = RVVBinaryDTypeKind::I32;
  RVVBinaryArithmeticKind arithmetic = RVVBinaryArithmeticKind::Add;
  llvm::StringRef dtypeID;
  unsigned elementBitWidth = 32;
  llvm::StringRef familyID;
  llvm::StringRef microkernelOpName;
  llvm::StringRef arithmeticOpName;
  llvm::StringRef arithmeticVerb;
  llvm::StringRef functionStem;
  llvm::StringRef headerGuardStem;
  llvm::StringRef resultCName;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
};

inline const RVVBinaryFamilyRecord &getI32VAddFamilyRegistrationRecord() {
  static const RVVBinaryFamilyRecord record{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Add,
      "i32",
      32,
      "i32-vadd",
      "tcrv_rvv.i32_vadd_microkernel",
      "tcrv_rvv.i32_add",
      "add",
      "i32_vadd",
      "I32_VADD",
      "sum_vec",
      "const int32_t *",
      "int32_t *"};
  return record;
}

inline const RVVBinaryFamilyRecord &getI32VSubFamilyRegistrationRecord() {
  static const RVVBinaryFamilyRecord record{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Sub,
      "i32",
      32,
      "i32-vsub",
      "tcrv_rvv.i32_vsub_microkernel",
      "tcrv_rvv.i32_sub",
      "subtract",
      "i32_vsub",
      "I32_VSUB",
      "difference_vec",
      "const int32_t *",
      "int32_t *"};
  return record;
}

inline const RVVBinaryFamilyRecord &getI32VMulFamilyRegistrationRecord() {
  static const RVVBinaryFamilyRecord record{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Mul,
      "i32",
      32,
      "i32-vmul",
      "tcrv_rvv.i32_vmul_microkernel",
      "tcrv_rvv.i32_mul",
      "multiply",
      "i32_vmul",
      "I32_VMUL",
      "product_vec",
      "const int32_t *",
      "int32_t *"};
  return record;
}

inline const RVVBinaryFamilyRecord &getI64VAddFamilyRegistrationRecord() {
  static const RVVBinaryFamilyRecord record{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Add,
      "i64",
      64,
      "i64-vadd",
      "tcrv_rvv.i64_vadd_microkernel",
      "tcrv_rvv.i64_add",
      "add",
      "i64_vadd",
      "I64_VADD",
      "sum_vec",
      "const int64_t *",
      "int64_t *"};
  return record;
}

inline const RVVBinaryFamilyRecord &getI64VSubFamilyRegistrationRecord() {
  static const RVVBinaryFamilyRecord record{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Sub,
      "i64",
      64,
      "i64-vsub",
      "tcrv_rvv.i64_vsub_microkernel",
      "tcrv_rvv.i64_sub",
      "subtract",
      "i64_vsub",
      "I64_VSUB",
      "difference_vec",
      "const int64_t *",
      "int64_t *"};
  return record;
}

inline const RVVBinaryFamilyRecord &getI64VMulFamilyRegistrationRecord() {
  static const RVVBinaryFamilyRecord record{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Mul,
      "i64",
      64,
      "i64-vmul",
      "tcrv_rvv.i64_vmul_microkernel",
      "tcrv_rvv.i64_mul",
      "multiply",
      "i64_vmul",
      "I64_VMUL",
      "product_vec",
      "const int64_t *",
      "int64_t *"};
  return record;
}

inline llvm::ArrayRef<const RVVBinaryFamilyRecord *>
getRVVBinaryFamilyRegistrationRecords() {
  static const RVVBinaryFamilyRecord *families[] = {
      &getI32VAddFamilyRegistrationRecord(), &getI32VSubFamilyRegistrationRecord(),
      &getI32VMulFamilyRegistrationRecord(), &getI64VAddFamilyRegistrationRecord(),
      &getI64VSubFamilyRegistrationRecord(), &getI64VMulFamilyRegistrationRecord()};
  return llvm::ArrayRef(families);
}

inline const RVVBinaryFamilyRecord *
lookupRVVBinaryFamilyRegistrationByID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const RVVBinaryFamilyRecord *record :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (record->familyID == familyID)
      return record;
  }
  return nullptr;
}

inline const RVVBinaryFamilyRecord *
lookupRVVBinaryFamilyRegistrationByRVVOperationName(
    llvm::StringRef rvvOperationName) {
  rvvOperationName = rvvOperationName.trim();
  for (const RVVBinaryFamilyRecord *record :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (record->arithmeticOpName == rvvOperationName)
      return record;
  }
  return nullptr;
}

inline llvm::ArrayRef<const RVVVectorShapeConfig *>
getRVVBinaryFamilyShapeConfigs(const RVVBinaryFamilyRecord &family) {
  switch (family.dtype) {
  case RVVBinaryDTypeKind::I32:
    return getFiniteI32VectorShapeConfigs();
  case RVVBinaryDTypeKind::I64:
    return getFiniteI64VectorShapeConfigs();
  }
  llvm_unreachable("unknown RVV binary dtype");
}

inline const RVVVectorShapeConfig *
lookupRVVBinaryFamilyShapeConfigByID(const RVVBinaryFamilyRecord &family,
                                     llvm::StringRef shapeID) {
  shapeID = shapeID.trim();
  for (const RVVVectorShapeConfig *shape :
       getRVVBinaryFamilyShapeConfigs(family)) {
    if (shape->shapeID == shapeID)
      return shape;
  }
  return nullptr;
}

class RVVBinaryRuntimeABIContract
    : public support::FiniteBinaryRuntimeABIContract {
public:
  explicit RVVBinaryRuntimeABIContract(const RVVBinaryFamilyRecord &family)
      : support::FiniteBinaryRuntimeABIContract(
            support::FiniteBinaryRuntimeABIContractSpec{
                family.familyID,
                family.constInputPointerCType,
                family.outputPointerCType,
                {},
                {},
                {}}),
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
  llvm_unreachable("unknown RVV binary family");
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVBinaryCallableRuntimeABIParameters(
    const RVVBinaryFamilyRecord &family,
    llvm::StringRef runtimeCountCName = "n") {
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
      runtimeCountCName.trim().empty() ? "n" : runtimeCountCName.trim(),
      "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVBinaryCallableRuntimeABIRoleRequirements(
    const RVVBinaryFamilyRecord &family) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> requirements;
  requirements.push_back(support::makeTargetExportABIRoleRequirement(
      family.constInputPointerCType,
      support::RuntimeABIParameterRole::LHSInputBuffer));
  requirements.push_back(support::makeTargetExportABIRoleRequirement(
      family.constInputPointerCType,
      support::RuntimeABIParameterRole::RHSInputBuffer));
  requirements.push_back(support::makeTargetExportABIRoleRequirement(
      family.outputPointerCType,
      support::RuntimeABIParameterRole::OutputBuffer));
  requirements.push_back(support::makeTargetExportABIRoleRequirement(
      "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return requirements;
}

inline llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3>
getRVVBinaryBufferMemWindowSpecs(const RVVBinaryFamilyRecord &family) {
  llvm::StringRef ownership = support::stringifyRuntimeABIParameterOwnership(
      support::RuntimeABIParameterOwnership::TargetExportABIOwned);
  llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3> specs;
  specs.push_back(support::RuntimeABIMemWindowSpec(
      "abi_lhs_input_buffer",
      support::RuntimeABIParameterRole::LHSInputBuffer,
      support::kRuntimeABIReadAccess, ownership, family.constInputPointerCType));
  specs.push_back(support::RuntimeABIMemWindowSpec(
      "abi_rhs_input_buffer",
      support::RuntimeABIParameterRole::RHSInputBuffer,
      support::kRuntimeABIReadAccess, ownership, family.constInputPointerCType));
  specs.push_back(support::RuntimeABIMemWindowSpec(
      "abi_output_buffer",
      support::RuntimeABIParameterRole::OutputBuffer,
      support::kRuntimeABIWriteAccess, ownership, family.outputPointerCType));
  return specs;
}

inline llvm::SmallVector<support::RuntimeABIParamSpec, 1>
getRVVBinaryRuntimeElementCountParamSpecs(
    const RVVBinaryFamilyRecord &family, llvm::StringRef cName = "n") {
  (void)family;
  llvm::StringRef trimmed = cName.trim();
  llvm::SmallVector<support::RuntimeABIParamSpec, 1> specs;
  specs.push_back(support::RuntimeABIParamSpec(
      "abi_runtime_element_count",
      support::RuntimeABIParameterRole::RuntimeElementCount,
      trimmed.empty() ? "n" : trimmed, "size_t",
      support::stringifyRuntimeABIParameterOwnership(
          support::RuntimeABIParameterOwnership::TargetExportABIOwned)));
  return specs;
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVBINARYFAMILY_H
