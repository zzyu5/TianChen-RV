#ifndef TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H
#define TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv::support {

enum class FiniteBinaryElementKind {
  I32,
  I64,
};

// Neutral frontend marker/ABI contract for the bounded linalg binary slice.
// It names accepted route markers and reusable ABI spellings only after the
// source linalg body has already determined the finite family. RVV/scalar
// lowering descriptors, arithmetic operators, selected vector shape, route
// ids, artifacts, and evidence remain plugin/target-owned.
struct FiniteBinaryFrontendContract {
  FiniteBinaryElementKind elementKind = FiniteBinaryElementKind::I32;
  llvm::StringRef dtypeID;
  unsigned elementBitWidth = 32;
  llvm::StringRef familyID;
  llvm::StringRef frontendLowering;
  llvm::StringRef scalarCType;
  llvm::StringRef constInputPointerCType;
  llvm::StringRef outputPointerCType;
};

inline const FiniteBinaryFrontendContract &
getI32VAddFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I32,
      "i32",
      32,
      "i32-vadd",
      "i32-vadd",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI32VSubFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I32,
      "i32",
      32,
      "i32-vsub",
      "i32-vsub",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI32VMulFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I32,
      "i32",
      32,
      "i32-vmul",
      "i32-vmul",
      "int32_t",
      "const int32_t *",
      "int32_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI64VAddFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I64,
      "i64",
      64,
      "i64-vadd",
      "i64-vadd",
      "int64_t",
      "const int64_t *",
      "int64_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI64VSubFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I64,
      "i64",
      64,
      "i64-vsub",
      "i64-vsub",
      "int64_t",
      "const int64_t *",
      "int64_t *"};
  return contract;
}

inline const FiniteBinaryFrontendContract &
getI64VMulFiniteBinaryFrontendContract() {
  static const FiniteBinaryFrontendContract contract{
      FiniteBinaryElementKind::I64,
      "i64",
      64,
      "i64-vmul",
      "i64-vmul",
      "int64_t",
      "const int64_t *",
      "int64_t *"};
  return contract;
}

inline llvm::ArrayRef<const FiniteBinaryFrontendContract *>
getFiniteBinaryFrontendContracts() {
  static const FiniteBinaryFrontendContract *families[] = {
      &getI32VAddFiniteBinaryFrontendContract(),
      &getI32VSubFiniteBinaryFrontendContract(),
      &getI32VMulFiniteBinaryFrontendContract(),
      &getI64VAddFiniteBinaryFrontendContract(),
      &getI64VSubFiniteBinaryFrontendContract(),
      &getI64VMulFiniteBinaryFrontendContract()};
  return llvm::ArrayRef(families);
}

inline const FiniteBinaryFrontendContract *
lookupFiniteBinaryFrontendContractByFamilyID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const FiniteBinaryFrontendContract *contract :
       getFiniteBinaryFrontendContracts())
    if (contract->familyID == familyID)
      return contract;
  return nullptr;
}

inline const FiniteBinaryFrontendContract *
lookupFiniteBinaryFrontendContractByMarker(llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  for (const FiniteBinaryFrontendContract *contract :
       getFiniteBinaryFrontendContracts())
    if (contract->frontendLowering == frontendLowering)
      return contract;
  return nullptr;
}

inline std::string formatFiniteBinaryFrontendLoweringMarkers() {
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::ArrayRef<const FiniteBinaryFrontendContract *> families =
      getFiniteBinaryFrontendContracts();
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
    const FiniteBinaryFrontendContract &contract) {
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> specs;
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_lhs_input_buffer", RuntimeABIParameterRole::LHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      contract.constInputPointerCType));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_rhs_input_buffer", RuntimeABIParameterRole::RHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      contract.constInputPointerCType));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_output_buffer", RuntimeABIParameterRole::OutputBuffer,
      kRuntimeABIWriteAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      contract.outputPointerCType));
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
    const FiniteBinaryFrontendContract &contract,
    llvm::StringRef cName = "n") {
  (void)contract;
  llvm::SmallVector<RuntimeABIParamSpec, 1> specs;
  specs.push_back(getFiniteBinaryFrontendRuntimeElementCountParamSpec(cName));
  return specs;
}

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_FINITEBINARYFRONTENDLOWERING_H
