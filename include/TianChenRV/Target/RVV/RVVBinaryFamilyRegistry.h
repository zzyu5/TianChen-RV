#ifndef TIANCHENRV_TARGET_RVV_RVVBINARYFAMILYREGISTRY_H
#define TIANCHENRV_TARGET_RVV_RVVBINARYFAMILYREGISTRY_H

#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

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
  unsigned elementBitWidth = 32;
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

inline const RVVBinaryFamilyDescriptor &getI32VAddFamilyDescriptor() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Add,
      "i32",
      32,
      "i32-vadd",
      "i32-vadd",
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
      "+",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI32VSubFamilyDescriptor() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Sub,
      "i32",
      32,
      "i32-vsub",
      "i32-vsub",
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
      "-",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI32VMulFamilyDescriptor() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I32,
      RVVBinaryArithmeticKind::Mul,
      "i32",
      32,
      "i32-vmul",
      "i32-vmul",
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
      "*",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return descriptor;
}

inline const RVVBinaryFamilyDescriptor &getI64VAddFamilyDescriptor() {
  static const RVVBinaryFamilyDescriptor descriptor{
      RVVBinaryDTypeKind::I64,
      RVVBinaryArithmeticKind::Add,
      "i64",
      64,
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

inline llvm::ArrayRef<const RVVBinaryFamilyDescriptor *>
getRVVBinaryFamilyDescriptors() {
  static const RVVBinaryFamilyDescriptor *families[] = {
      &getI32VAddFamilyDescriptor(), &getI32VSubFamilyDescriptor(),
      &getI32VMulFamilyDescriptor(), &getI64VAddFamilyDescriptor()};
  return llvm::ArrayRef(families);
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyByID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyDescriptors()) {
    if (descriptor->familyID == familyID)
      return descriptor;
  }
  return nullptr;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyByFrontendLowering(llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyDescriptors()) {
    if (descriptor->frontendLowering == frontendLowering)
      return descriptor;
  }
  return nullptr;
}

inline const RVVBinaryFamilyDescriptor *
lookupRVVBinaryFamilyByLoweringDescriptor(llvm::StringRef loweringDescriptor) {
  loweringDescriptor = loweringDescriptor.trim();
  for (const RVVBinaryFamilyDescriptor *descriptor :
       getRVVBinaryFamilyDescriptors()) {
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
