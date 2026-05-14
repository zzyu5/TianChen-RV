#ifndef TIANCHENRV_TARGET_RVV_RVVBINARYFAMILYREGISTRY_H
#define TIANCHENRV_TARGET_RVV_RVVBINARYFAMILYREGISTRY_H

#include "TianChenRV/Support/FiniteBinaryFrontendLowering.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
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

// Finite RVV binary family registration metadata. This table is allowed to
// register bounded routes and compatibility names, but typed selected plans,
// exec-IR ABI boundaries, and EmitC route metadata remain production
// authority for source/ABI emission.
struct RVVBinaryFamilyDescriptor {
  RVVBinaryDTypeKind dtype = RVVBinaryDTypeKind::I32;
  RVVBinaryArithmeticKind arithmetic = RVVBinaryArithmeticKind::Add;
  llvm::StringRef dtypeID;
  unsigned elementBitWidth = 32;
  llvm::StringRef familyID;
  llvm::StringRef frontendLowering;
  llvm::StringRef sourceArithmeticOpName;
  llvm::StringRef dynamicVectorSourceKind;
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
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
  const support::FiniteBinaryFrontendContract *frontendContract = nullptr;
};

inline const RVVBinaryFamilyDescriptor &getI32VAddFamilyRegistrationRecord() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Add,
      "i32",
      32,
      "i32-vadd",
      "i32-vadd",
      "arith.addi",
      support::kFrontendDynamicVectorI32VAddSourceKind,
      "i32-vadd-microkernel.v1",
      "finite RVV i32-vadd lowering descriptor",
      "tcrv_rvv.i32_vadd_microkernel",
      "tcrv_rvv.i32_add",
      "add",
      "i32_vadd",
      "I32_VADD",
      "__riscv_vadd_vv_",
      "sum_vec",
      "rvv-explicit-i32-vadd-microkernel-c-source",
      "tcrv-export-rvv-microkernel-c",
      "tcrv-export-rvv-microkernel-header",
      "tcrv-export-rvv-microkernel-object",
      "rvv-i32-vadd-runtime-callable-c-abi.v1",
      "rvv-runtime-callable-c-abi",
      "rvv-i32-vadd-runtime-callable-c-function.v1",
      "runtime-callable-i32-vadd-function",
      "rvv-i32-vadd-microkernel-external-abi.v1",
      "const int32_t *",
      "int32_t *",
      &support::getI32VAddFiniteBinaryFrontendContract()};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI32VSubFamilyRegistrationRecord() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Sub,
      "i32",
      32,
      "i32-vsub",
      "i32-vsub",
      "arith.subi",
      support::kFrontendDynamicVectorI32VSubSourceKind,
      "i32-vsub-microkernel.v1",
      "finite RVV i32-vsub lowering descriptor",
      "tcrv_rvv.i32_vsub_microkernel",
      "tcrv_rvv.i32_sub",
      "subtract",
      "i32_vsub",
      "I32_VSUB",
      "__riscv_vsub_vv_",
      "difference_vec",
      "rvv-explicit-i32-vsub-microkernel-c-source",
      "tcrv-export-rvv-i32-vsub-microkernel-c",
      "tcrv-export-rvv-i32-vsub-microkernel-header",
      "tcrv-export-rvv-i32-vsub-microkernel-object",
      "rvv-i32-vsub-runtime-callable-c-abi.v1",
      "rvv-runtime-callable-c-abi",
      "rvv-i32-vsub-runtime-callable-c-function.v1",
      "runtime-callable-i32-vsub-function",
      "rvv-i32-vsub-microkernel-external-abi.v1",
      "const int32_t *",
      "int32_t *",
      &support::getI32VSubFiniteBinaryFrontendContract()};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI32VMulFamilyRegistrationRecord() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Mul,
      "i32",
      32,
      "i32-vmul",
      "i32-vmul",
      "arith.muli",
      support::kFrontendDynamicVectorI32VMulSourceKind,
      "i32-vmul-microkernel.v1",
      "finite RVV i32-vmul lowering descriptor",
      "tcrv_rvv.i32_vmul_microkernel",
      "tcrv_rvv.i32_mul",
      "multiply",
      "i32_vmul",
      "I32_VMUL",
      "__riscv_vmul_vv_",
      "product_vec",
      "rvv-explicit-i32-vmul-microkernel-c-source",
      "tcrv-export-rvv-i32-vmul-microkernel-c",
      "tcrv-export-rvv-i32-vmul-microkernel-header",
      "tcrv-export-rvv-i32-vmul-microkernel-object",
      "rvv-i32-vmul-runtime-callable-c-abi.v1",
      "rvv-runtime-callable-c-abi",
      "rvv-i32-vmul-runtime-callable-c-function.v1",
      "runtime-callable-i32-vmul-function",
      "rvv-i32-vmul-microkernel-external-abi.v1",
      "const int32_t *",
      "int32_t *",
      &support::getI32VMulFiniteBinaryFrontendContract()};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI64VAddFamilyRegistrationRecord() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Add,
      "i64",
      64,
      "i64-vadd",
      "i64-vadd",
      "arith.addi",
      "",
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
      "const int64_t *",
      "int64_t *",
      &support::getI64VAddFiniteBinaryFrontendContract()};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI64VSubFamilyRegistrationRecord() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Sub,
      "i64",
      64,
      "i64-vsub",
      "i64-vsub",
      "arith.subi",
      "",
      "i64-vsub-microkernel.v1",
      "finite i64-vsub lowering descriptor",
      "tcrv_rvv.i64_vsub_microkernel",
      "tcrv_rvv.i64_sub",
      "subtract",
      "i64_vsub",
      "I64_VSUB",
      "__riscv_vsub_vv_",
      "difference_vec",
      "rvv-explicit-i64-vsub-microkernel-c-source",
      "tcrv-export-rvv-i64-vsub-microkernel-c",
      "tcrv-export-rvv-i64-vsub-microkernel-header",
      "tcrv-export-rvv-i64-vsub-microkernel-object",
      "rvv-i64-vsub-runtime-callable-c-abi.v1",
      "rvv-runtime-callable-c-abi",
      "rvv-i64-vsub-runtime-callable-c-function.v1",
      "runtime-callable-i64-vsub-function",
      "rvv-i64-vsub-microkernel-external-abi.v1",
      "const int64_t *",
      "int64_t *",
      &support::getI64VSubFiniteBinaryFrontendContract()};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI64VMulFamilyRegistrationRecord() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Mul,
      "i64",
      64,
      "i64-vmul",
      "i64-vmul",
      "arith.muli",
      "",
      "i64-vmul-microkernel.v1",
      "finite i64-vmul lowering descriptor",
      "tcrv_rvv.i64_vmul_microkernel",
      "tcrv_rvv.i64_mul",
      "multiply",
      "i64_vmul",
      "I64_VMUL",
      "__riscv_vmul_vv_",
      "product_vec",
      "rvv-explicit-i64-vmul-microkernel-c-source",
      "tcrv-export-rvv-i64-vmul-microkernel-c",
      "tcrv-export-rvv-i64-vmul-microkernel-header",
      "tcrv-export-rvv-i64-vmul-microkernel-object",
      "rvv-i64-vmul-runtime-callable-c-abi.v1",
      "rvv-runtime-callable-c-abi",
      "rvv-i64-vmul-runtime-callable-c-function.v1",
      "runtime-callable-i64-vmul-function",
      "rvv-i64-vmul-microkernel-external-abi.v1",
      "const int64_t *",
      "int64_t *",
      &support::getI64VMulFiniteBinaryFrontendContract()};
  return descriptor;
}

inline llvm::ArrayRef<const RVVBinaryFamilyDescriptor *>
getRVVBinaryFamilyRegistrationRecords() {
  static const RVVBinaryFamilyDescriptor *families[] = {
      &getI32VAddFamilyRegistrationRecord(), &getI32VSubFamilyRegistrationRecord(),
      &getI32VMulFamilyRegistrationRecord(), &getI64VAddFamilyRegistrationRecord(),
      &getI64VSubFamilyRegistrationRecord(), &getI64VMulFamilyRegistrationRecord()};
  return llvm::ArrayRef(families);
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (descriptor->familyID == familyID)
      return descriptor;
  }
  return nullptr;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByFrontendLowering(llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (descriptor->frontendLowering == frontendLowering)
      return descriptor;
  }
  return nullptr;
}

inline std::optional<RVVBinaryDTypeKind>
convertFrontendElementKindToRVVDTypeKind(
    support::FiniteBinaryElementKind elementKind) {
  switch (elementKind) {
  case support::FiniteBinaryElementKind::I32:
    return RVVBinaryDTypeKind::I32;
  case support::FiniteBinaryElementKind::I64:
    return RVVBinaryDTypeKind::I64;
  }
  return std::nullopt;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByFrontendContract(
    const support::FiniteBinaryFrontendContract &contract) {
  const RVVBinaryFamilyDescriptor *family =
      lookupRVVBinaryFamilyRegistrationByID(contract.familyID);
  if (!family || !family->frontendContract ||
      family->frontendContract->frontendLowering != contract.frontendLowering ||
      family->frontendContract->elementKind != contract.elementKind ||
      family->frontendContract->elementBitWidth != contract.elementBitWidth ||
      family->frontendContract->constInputPointerCType !=
          contract.constInputPointerCType ||
      family->frontendContract->outputPointerCType !=
          contract.outputPointerCType)
    return nullptr;
  return family;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByFrontendSource(
    support::FiniteBinaryElementKind elementKind,
    llvm::StringRef sourceArithmeticOpName) {
  std::optional<RVVBinaryDTypeKind> dtype =
      convertFrontendElementKindToRVVDTypeKind(elementKind);
  if (!dtype)
    return nullptr;
  sourceArithmeticOpName = sourceArithmeticOpName.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (descriptor->dtype == *dtype &&
        descriptor->sourceArithmeticOpName == sourceArithmeticOpName)
      return descriptor;
  }
  return nullptr;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByRVVOperationName(
    llvm::StringRef rvvOperationName) {
  rvvOperationName = rvvOperationName.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (descriptor->arithmeticOpName == rvvOperationName)
      return descriptor;
  }
  return nullptr;
}

inline bool isRVVBinaryFamilyAcceptedByDynamicVectorSource(
    const RVVBinaryFamilyDescriptor &family) {
  return !family.dynamicVectorSourceKind.empty();
}

inline llvm::StringRef getRVVDynamicVectorSourceKindForFamily(
    const RVVBinaryFamilyDescriptor &family) {
  return family.dynamicVectorSourceKind;
}

inline std::string formatRVVBinaryFrontendLoweringMarkers() {
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::ArrayRef<const RVVBinaryFamilyDescriptor *> families =
      getRVVBinaryFamilyRegistrationRecords();
  for (auto [index, family] : llvm::enumerate(families)) {
    if (index != 0) {
      if (index + 1 == families.size())
        stream << ", or ";
      else
        stream << ", ";
    }
    stream << "'" << family->frontendLowering << "'";
  }
  stream.flush();
  return text;
}

inline std::string formatRVVDynamicVectorFrontendLowerings() {
  std::string text;
  llvm::raw_string_ostream stream(text);
  bool first = true;
  unsigned remaining = 0;
  for (const RVVBinaryFamilyDescriptor *family :
       getRVVBinaryFamilyRegistrationRecords())
    if (isRVVBinaryFamilyAcceptedByDynamicVectorSource(*family))
      ++remaining;

  for (const RVVBinaryFamilyDescriptor *family :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (!isRVVBinaryFamilyAcceptedByDynamicVectorSource(*family))
      continue;
    if (!first) {
      if (remaining == 1)
        stream << " or ";
      else
        stream << ", ";
    }
    stream << "'" << family->frontendLowering << "'";
    first = false;
    --remaining;
  }
  stream.flush();
  return text;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(llvm::StringRef loweringDescriptor) {
  loweringDescriptor = loweringDescriptor.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyRegistrationRecords()) {
    if (descriptor->loweringDescriptor == loweringDescriptor)
      return descriptor;
  }
  return nullptr;
}

inline llvm::ArrayRef<const RVVVectorShapeConfig *>
getRVVBinaryFamilyShapeConfigs(const RVVBinaryFamilyDescriptor &family) {
  switch (family.dtype) {
  case RVVBinaryDTypeKind::I32:
    return getFiniteI32VectorShapeConfigs();
  case RVVBinaryDTypeKind::I64:
    return getFiniteI64VectorShapeConfigs();
  }
  llvm_unreachable("unknown RVV binary dtype");
}

inline const RVVVectorShapeConfig *
lookupRVVBinaryFamilyShapeConfigByID(const RVVBinaryFamilyDescriptor &family,
                                     llvm::StringRef shapeID) {
  shapeID = shapeID.trim();
  for (const RVVVectorShapeConfig *shape :
       getRVVBinaryFamilyShapeConfigs(family)) {
    if (shape->shapeID == shapeID)
      return shape;
  }
  return nullptr;
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVBINARYFAMILYREGISTRY_H
