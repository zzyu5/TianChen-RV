#ifndef TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H
#define TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::support {

enum class FiniteBinaryElementKind {
  I32,
  I64,
};

enum class FiniteBinaryArithmeticKind {
  Add,
  Sub,
  Mul,
};

// Neutral frontend contract for the bounded linalg binary slice. It names only
// finite source/ABI facts; RVV/scalar lowering, selected vector shape, route
// ids, artifacts, and evidence remain plugin/target-owned.
struct FiniteBinaryFrontendLoweringDescriptor {
  FiniteBinaryElementKind elementKind = FiniteBinaryElementKind::I32;
  FiniteBinaryArithmeticKind arithmetic = FiniteBinaryArithmeticKind::Add;
  llvm::StringRef dtypeID;
  unsigned elementBitWidth = 32;
  llvm::StringRef familyID;
  llvm::StringRef frontendLowering;
  llvm::StringRef loweringDescriptor;
  llvm::StringRef arithmeticVerb;
  llvm::StringRef cOperator;
  llvm::StringRef scalarCType;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
};

inline const FiniteBinaryFrontendLoweringDescriptor &
getI32VAddFiniteBinaryFrontendLoweringDescriptor() {
  static const FiniteBinaryFrontendLoweringDescriptor descriptor{
      FiniteBinaryElementKind::I32,
      FiniteBinaryArithmeticKind::Add,
      "i32",
      32,
      "i32-vadd",
      "i32-vadd",
      "i32-vadd-microkernel.v1",
      "add",
      "+",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return descriptor;
}

inline const FiniteBinaryFrontendLoweringDescriptor &
getI32VSubFiniteBinaryFrontendLoweringDescriptor() {
  static const FiniteBinaryFrontendLoweringDescriptor descriptor{
      FiniteBinaryElementKind::I32,
      FiniteBinaryArithmeticKind::Sub,
      "i32",
      32,
      "i32-vsub",
      "i32-vsub",
      "i32-vsub-microkernel.v1",
      "subtract",
      "-",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return descriptor;
}

inline const FiniteBinaryFrontendLoweringDescriptor &
getI32VMulFiniteBinaryFrontendLoweringDescriptor() {
  static const FiniteBinaryFrontendLoweringDescriptor descriptor{
      FiniteBinaryElementKind::I32,
      FiniteBinaryArithmeticKind::Mul,
      "i32",
      32,
      "i32-vmul",
      "i32-vmul",
      "i32-vmul-microkernel.v1",
      "multiply",
      "*",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return descriptor;
}

inline const FiniteBinaryFrontendLoweringDescriptor &
getI64VAddFiniteBinaryFrontendLoweringDescriptor() {
  static const FiniteBinaryFrontendLoweringDescriptor descriptor{
      FiniteBinaryElementKind::I64,
      FiniteBinaryArithmeticKind::Add,
      "i64",
      64,
      "i64-vadd",
      "i64-vadd",
      "i64-vadd-microkernel.v1",
      "add",
      "+",
      "int64_t",
      "const int64_t *",
      "int64_t *"};
  return descriptor;
}

inline const FiniteBinaryFrontendLoweringDescriptor &
getI64VSubFiniteBinaryFrontendLoweringDescriptor() {
  static const FiniteBinaryFrontendLoweringDescriptor descriptor{
      FiniteBinaryElementKind::I64,
      FiniteBinaryArithmeticKind::Sub,
      "i64",
      64,
      "i64-vsub",
      "i64-vsub",
      "i64-vsub-microkernel.v1",
      "subtract",
      "-",
      "int64_t",
      "const int64_t *",
      "int64_t *"};
  return descriptor;
}

inline const FiniteBinaryFrontendLoweringDescriptor &
getI64VMulFiniteBinaryFrontendLoweringDescriptor() {
  static const FiniteBinaryFrontendLoweringDescriptor descriptor{
      FiniteBinaryElementKind::I64,
      FiniteBinaryArithmeticKind::Mul,
      "i64",
      64,
      "i64-vmul",
      "i64-vmul",
      "i64-vmul-microkernel.v1",
      "multiply",
      "*",
      "int64_t",
      "const int64_t *",
      "int64_t *"};
  return descriptor;
}

inline llvm::ArrayRef<const FiniteBinaryFrontendLoweringDescriptor *>
getFiniteBinaryFrontendLoweringDescriptors() {
  static const FiniteBinaryFrontendLoweringDescriptor *families[] = {
      &getI32VAddFiniteBinaryFrontendLoweringDescriptor(),
      &getI32VSubFiniteBinaryFrontendLoweringDescriptor(),
      &getI32VMulFiniteBinaryFrontendLoweringDescriptor(),
      &getI64VAddFiniteBinaryFrontendLoweringDescriptor(),
      &getI64VSubFiniteBinaryFrontendLoweringDescriptor(),
      &getI64VMulFiniteBinaryFrontendLoweringDescriptor()};
  return llvm::ArrayRef(families);
}

inline const FiniteBinaryFrontendLoweringDescriptor *
lookupFiniteBinaryFrontendLoweringByFamilyID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const FiniteBinaryFrontendLoweringDescriptor *descriptor :
       getFiniteBinaryFrontendLoweringDescriptors())
    if (descriptor->familyID == familyID)
      return descriptor;
  return nullptr;
}

inline const FiniteBinaryFrontendLoweringDescriptor *
lookupFiniteBinaryFrontendLoweringByMarker(llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  for (const FiniteBinaryFrontendLoweringDescriptor *descriptor :
       getFiniteBinaryFrontendLoweringDescriptors())
    if (descriptor->frontendLowering == frontendLowering)
      return descriptor;
  return nullptr;
}

inline const FiniteBinaryFrontendLoweringDescriptor *
lookupFiniteBinaryFrontendLoweringByDescriptor(
    llvm::StringRef loweringDescriptor) {
  loweringDescriptor = loweringDescriptor.trim();
  for (const FiniteBinaryFrontendLoweringDescriptor *descriptor :
       getFiniteBinaryFrontendLoweringDescriptors())
    if (descriptor->loweringDescriptor == loweringDescriptor)
      return descriptor;
  return nullptr;
}

inline std::string formatFiniteBinaryFrontendLoweringMarkers() {
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::ArrayRef<const FiniteBinaryFrontendLoweringDescriptor *> families =
      getFiniteBinaryFrontendLoweringDescriptors();
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

inline llvm::SmallVector<RuntimeABIMemWindowSpec, 3>
getFiniteBinaryFrontendBufferMemWindowSpecs(
    const FiniteBinaryFrontendLoweringDescriptor &descriptor) {
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> specs;
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_lhs_input_buffer", RuntimeABIParameterRole::LHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      descriptor.constInputPointerCType));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_rhs_input_buffer", RuntimeABIParameterRole::RHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      descriptor.constInputPointerCType));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_output_buffer", RuntimeABIParameterRole::OutputBuffer,
      kRuntimeABIWriteAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      descriptor.outputPointerCType));
  return specs;
}

inline RuntimeABIParamSpec getFiniteBinaryFrontendRuntimeElementCountParamSpec(
    llvm::StringRef cName = "n") {
  return RuntimeABIParamSpec(
      "abi_runtime_element_count",
      RuntimeABIParameterRole::RuntimeElementCount, cName, "size_t",
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned));
}

inline llvm::SmallVector<RuntimeABIParamSpec, 1>
getFiniteBinaryFrontendRuntimeElementCountParamSpecs(
    const FiniteBinaryFrontendLoweringDescriptor &descriptor,
    llvm::StringRef cName = "n") {
  (void)descriptor;
  llvm::SmallVector<RuntimeABIParamSpec, 1> specs;
  specs.push_back(getFiniteBinaryFrontendRuntimeElementCountParamSpec(cName));
  return specs;
}

inline llvm::StringRef stringifyFiniteBinaryArithmeticKind(
    FiniteBinaryArithmeticKind arithmetic) {
  switch (arithmetic) {
  case FiniteBinaryArithmeticKind::Add:
    return "add";
  case FiniteBinaryArithmeticKind::Sub:
    return "sub";
  case FiniteBinaryArithmeticKind::Mul:
    return "mul";
  }
  llvm_unreachable("unknown finite binary arithmetic kind");
}

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H
