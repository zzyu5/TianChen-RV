#ifndef TIANCHENRV_TARGET_I32BINARYFAMILYREGISTRY_H
#define TIANCHENRV_TARGET_I32BINARYFAMILYREGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace tianchenrv::target::i32_binary {

enum class I32BinaryFamilyKind {
  Add,
  Sub,
  Mul,
};

struct RVVI32MicrokernelFamilyDescriptor {
  I32BinaryFamilyKind kind;
  llvm::StringRef microkernelOpName;
  llvm::StringRef arithmeticOpName;
  llvm::StringRef arithmeticVerb;
  llvm::StringRef functionStem;
  llvm::StringRef headerGuardStem;
  llvm::StringRef intrinsicName;
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
};

struct ScalarI32MicrokernelFamilyDescriptor {
  I32BinaryFamilyKind kind;
  llvm::StringRef microkernelOpName;
  llvm::StringRef operationNoun;
  llvm::StringRef functionStem;
  llvm::StringRef headerGuardStem;
  llvm::StringRef descriptor;
  llvm::StringRef emissionKind;
  llvm::StringRef routeID;
  llvm::StringRef headerRouteID;
  llvm::StringRef objectRouteID;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef cOperator;
};

struct DispatchI32FamilyDescriptor {
  I32BinaryFamilyKind kind;
  llvm::StringRef diagnosticName;
  llvm::StringRef operationNoun;
  llvm::StringRef functionStem;
  llvm::StringRef headerGuardStem;
  llvm::StringRef cOperator;
  llvm::StringRef selfCheckSuccessMarker;
  llvm::StringRef rvvRouteID;
  llvm::StringRef rvvEmissionKind;
  llvm::StringRef rvvRuntimeABI;
  llvm::StringRef rvvRuntimeABIKind;
  llvm::StringRef rvvRuntimeABIName;
  llvm::StringRef rvvRuntimeGlueRole;
  llvm::StringRef scalarRouteID;
  llvm::StringRef scalarEmissionKind;
  llvm::StringRef scalarRuntimeABI;
  llvm::StringRef scalarRuntimeABIKind;
  llvm::StringRef scalarRuntimeABIName;
  llvm::StringRef scalarRuntimeGlueRole;
  llvm::StringRef dispatchSourceRouteID;
  llvm::StringRef dispatchHeaderRouteID;
  llvm::StringRef dispatchObjectRouteID;
  llvm::StringRef dispatchRuntimeABIKind;
  llvm::StringRef dispatchRuntimeABIName;
  llvm::StringRef dispatchExternalABIComponentGroup;
};

struct I32BinaryFamilyDescriptor {
  I32BinaryFamilyKind kind;
  llvm::StringRef familyID;
  llvm::StringRef frontendLowering;
  llvm::StringRef loweringDescriptor;
  llvm::StringRef descriptorNoun;
  RVVI32MicrokernelFamilyDescriptor rvv;
  ScalarI32MicrokernelFamilyDescriptor scalar;
  DispatchI32FamilyDescriptor dispatch;
};

inline const I32BinaryFamilyDescriptor &getI32VAddFamilyDescriptor() {
  static const I32BinaryFamilyDescriptor descriptor{
      I32BinaryFamilyKind::Add,
      "i32-vadd",
      "i32-vadd",
      "i32-vadd-microkernel.v1",
      "finite i32-vadd lowering descriptor",
      {I32BinaryFamilyKind::Add,
       "tcrv_rvv.i32_vadd_microkernel",
       "tcrv_rvv.i32_add",
       "add",
       "i32_vadd",
       "I32_VADD",
       "__riscv_vadd_vv_i32m1",
       "sum_vec",
       "rvv-explicit-i32-vadd-microkernel-c-source",
       "tcrv-export-rvv-microkernel-c",
       "tcrv-export-rvv-microkernel-header",
       "tcrv-export-rvv-microkernel-object",
       "rvv-i32-vadd-runtime-callable-c-abi.v1",
       "rvv-runtime-callable-c-abi",
       "rvv-i32-vadd-runtime-callable-c-function.v1",
       "runtime-callable-i32-vadd-function",
       "rvv-i32-vadd-microkernel-external-abi.v1"},
      {I32BinaryFamilyKind::Add,
       "tcrv_scalar.i32_vadd_microkernel",
       "i32 vector-add",
       "i32_vadd",
       "I32_VADD",
       "i32-vadd-microkernel.v1",
       "scalar-explicit-i32-vadd-microkernel-c-source",
       "tcrv-export-scalar-microkernel-c",
       "tcrv-export-scalar-microkernel-header",
       "tcrv-export-scalar-microkernel-object",
       "scalar-i32-vadd-runtime-callable-c-abi.v1",
       "scalar-runtime-callable-c-abi",
       "scalar-i32-vadd-runtime-callable-c-function.v1",
       "runtime-callable-i32-vadd-fallback-function",
       "+"},
      {I32BinaryFamilyKind::Add,
       "i32-vadd",
       "i32 vector-add",
       "i32_vadd",
       "I32_VADD",
       "+",
       "tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok",
       "tcrv-export-rvv-microkernel-c",
       "rvv-explicit-i32-vadd-microkernel-c-source",
       "rvv-i32-vadd-runtime-callable-c-abi.v1",
       "rvv-runtime-callable-c-abi",
       "rvv-i32-vadd-runtime-callable-c-function.v1",
       "runtime-callable-i32-vadd-function",
       "tcrv-export-scalar-microkernel-c",
       "scalar-explicit-i32-vadd-microkernel-c-source",
       "scalar-i32-vadd-runtime-callable-c-abi.v1",
       "scalar-runtime-callable-c-abi",
       "scalar-i32-vadd-runtime-callable-c-function.v1",
       "runtime-callable-i32-vadd-fallback-function",
       "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
       "tcrv-export-rvv-scalar-i32-vadd-dispatch-header",
       "tcrv-export-rvv-scalar-i32-vadd-dispatch-object",
       "rvv-scalar-dispatch-runtime-callable-c-abi",
       "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1",
       "rvv-scalar-i32-vadd-dispatch-external-abi.v1"}};
  return descriptor;
}

inline const I32BinaryFamilyDescriptor &getI32VSubFamilyDescriptor() {
  static const I32BinaryFamilyDescriptor descriptor{
      I32BinaryFamilyKind::Sub,
      "i32-vsub",
      "i32-vsub",
      "i32-vsub-microkernel.v1",
      "finite i32-vsub lowering descriptor",
      {I32BinaryFamilyKind::Sub,
       "tcrv_rvv.i32_vsub_microkernel",
       "tcrv_rvv.i32_sub",
       "subtract",
       "i32_vsub",
       "I32_VSUB",
       "__riscv_vsub_vv_i32m1",
       "difference_vec",
       "rvv-explicit-i32-vsub-microkernel-c-source",
       "tcrv-export-rvv-i32-vsub-microkernel-c",
       "tcrv-export-rvv-i32-vsub-microkernel-header",
       "tcrv-export-rvv-i32-vsub-microkernel-object",
       "rvv-i32-vsub-runtime-callable-c-abi.v1",
       "rvv-runtime-callable-c-abi",
       "rvv-i32-vsub-runtime-callable-c-function.v1",
       "runtime-callable-i32-vsub-function",
       "rvv-i32-vsub-microkernel-external-abi.v1"},
      {I32BinaryFamilyKind::Sub,
       "tcrv_scalar.i32_vsub_microkernel",
       "i32 vector-subtract",
       "i32_vsub",
       "I32_VSUB",
       "i32-vsub-microkernel.v1",
       "scalar-explicit-i32-vsub-microkernel-c-source",
       "tcrv-export-scalar-i32-vsub-microkernel-c",
       "tcrv-export-scalar-microkernel-header",
       "tcrv-export-scalar-microkernel-object",
       "scalar-i32-vsub-runtime-callable-c-abi.v1",
       "scalar-runtime-callable-c-abi",
       "scalar-i32-vsub-runtime-callable-c-function.v1",
       "runtime-callable-i32-vsub-fallback-function",
       "-"},
      {I32BinaryFamilyKind::Sub,
       "i32-vsub",
       "i32 vector-subtract",
       "i32_vsub",
       "I32_VSUB",
       "-",
       "tcrv_rvv_scalar_i32_vsub_dispatch_self_check_ok",
       "tcrv-export-rvv-i32-vsub-microkernel-c",
       "rvv-explicit-i32-vsub-microkernel-c-source",
       "rvv-i32-vsub-runtime-callable-c-abi.v1",
       "rvv-runtime-callable-c-abi",
       "rvv-i32-vsub-runtime-callable-c-function.v1",
       "runtime-callable-i32-vsub-function",
       "tcrv-export-scalar-i32-vsub-microkernel-c",
       "scalar-explicit-i32-vsub-microkernel-c-source",
       "scalar-i32-vsub-runtime-callable-c-abi.v1",
       "scalar-runtime-callable-c-abi",
       "scalar-i32-vsub-runtime-callable-c-function.v1",
       "runtime-callable-i32-vsub-fallback-function",
       "tcrv-export-rvv-scalar-i32-vsub-dispatch-c",
       "tcrv-export-rvv-scalar-i32-vsub-dispatch-header",
       "tcrv-export-rvv-scalar-i32-vsub-dispatch-object",
       "rvv-scalar-dispatch-runtime-callable-c-abi",
       "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1",
       "rvv-scalar-i32-vsub-dispatch-external-abi.v1"}};
  return descriptor;
}

inline const I32BinaryFamilyDescriptor &getI32VMulFamilyDescriptor() {
  static const I32BinaryFamilyDescriptor descriptor{
      I32BinaryFamilyKind::Mul,
      "i32-vmul",
      "i32-vmul",
      "i32-vmul-microkernel.v1",
      "finite i32-vmul lowering descriptor",
      {I32BinaryFamilyKind::Mul,
       "tcrv_rvv.i32_vmul_microkernel",
       "tcrv_rvv.i32_mul",
       "multiply",
       "i32_vmul",
       "I32_VMUL",
       "__riscv_vmul_vv_i32m1",
       "product_vec",
       "rvv-explicit-i32-vmul-microkernel-c-source",
       "tcrv-export-rvv-i32-vmul-microkernel-c",
       "tcrv-export-rvv-i32-vmul-microkernel-header",
       "tcrv-export-rvv-i32-vmul-microkernel-object",
       "rvv-i32-vmul-runtime-callable-c-abi.v1",
       "rvv-runtime-callable-c-abi",
       "rvv-i32-vmul-runtime-callable-c-function.v1",
       "runtime-callable-i32-vmul-function",
       "rvv-i32-vmul-microkernel-external-abi.v1"},
      {I32BinaryFamilyKind::Mul,
       "tcrv_scalar.i32_vmul_microkernel",
       "i32 vector-multiply",
       "i32_vmul",
       "I32_VMUL",
       "i32-vmul-microkernel.v1",
       "scalar-explicit-i32-vmul-microkernel-c-source",
       "tcrv-export-scalar-i32-vmul-microkernel-c",
       "tcrv-export-scalar-microkernel-header",
       "tcrv-export-scalar-microkernel-object",
       "scalar-i32-vmul-runtime-callable-c-abi.v1",
       "scalar-runtime-callable-c-abi",
       "scalar-i32-vmul-runtime-callable-c-function.v1",
       "runtime-callable-i32-vmul-fallback-function",
       "*"},
      {I32BinaryFamilyKind::Mul,
       "i32-vmul",
       "i32 vector-multiply",
       "i32_vmul",
       "I32_VMUL",
       "*",
       "tcrv_rvv_scalar_i32_vmul_dispatch_self_check_ok",
       "tcrv-export-rvv-i32-vmul-microkernel-c",
       "rvv-explicit-i32-vmul-microkernel-c-source",
       "rvv-i32-vmul-runtime-callable-c-abi.v1",
       "rvv-runtime-callable-c-abi",
       "rvv-i32-vmul-runtime-callable-c-function.v1",
       "runtime-callable-i32-vmul-function",
       "tcrv-export-scalar-i32-vmul-microkernel-c",
       "scalar-explicit-i32-vmul-microkernel-c-source",
       "scalar-i32-vmul-runtime-callable-c-abi.v1",
       "scalar-runtime-callable-c-abi",
       "scalar-i32-vmul-runtime-callable-c-function.v1",
       "runtime-callable-i32-vmul-fallback-function",
       "tcrv-export-rvv-scalar-i32-vmul-dispatch-c",
       "tcrv-export-rvv-scalar-i32-vmul-dispatch-header",
       "tcrv-export-rvv-scalar-i32-vmul-dispatch-object",
       "rvv-scalar-dispatch-runtime-callable-c-abi",
       "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1",
       "rvv-scalar-i32-vmul-dispatch-external-abi.v1"}};
  return descriptor;
}

inline llvm::ArrayRef<const I32BinaryFamilyDescriptor *>
getI32BinaryFamilyDescriptors() {
  static const I32BinaryFamilyDescriptor *families[] = {
      &getI32VAddFamilyDescriptor(), &getI32VSubFamilyDescriptor(),
      &getI32VMulFamilyDescriptor()};
  return llvm::ArrayRef(families);
}

inline const I32BinaryFamilyDescriptor *
lookupI32BinaryFamilyByID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const I32BinaryFamilyDescriptor *descriptor :
       getI32BinaryFamilyDescriptors()) {
    if (descriptor->familyID == familyID)
      return descriptor;
  }
  return nullptr;
}

inline const I32BinaryFamilyDescriptor *
lookupI32BinaryFamilyByFrontendLowering(llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  for (const I32BinaryFamilyDescriptor *descriptor :
       getI32BinaryFamilyDescriptors()) {
    if (descriptor->frontendLowering == frontendLowering)
      return descriptor;
  }
  return nullptr;
}

inline const I32BinaryFamilyDescriptor *
lookupI32BinaryFamilyByLoweringDescriptor(llvm::StringRef loweringDescriptor) {
  loweringDescriptor = loweringDescriptor.trim();
  for (const I32BinaryFamilyDescriptor *descriptor :
       getI32BinaryFamilyDescriptors()) {
    if (descriptor->loweringDescriptor == loweringDescriptor)
      return descriptor;
  }
  return nullptr;
}

} // namespace tianchenrv::target::i32_binary

#endif // TIANCHENRV_TARGET_I32BINARYFAMILYREGISTRY_H
