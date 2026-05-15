#ifndef TIANCHENRV_TARGET_RVVSCALARBINARYFAMILY_H
#define TIANCHENRV_TARGET_RVVSCALARBINARYFAMILY_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamily.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace tianchenrv::target::rvv_scalar {

// Scalar typed-family metadata derived from the finite RVV family table. This
// surface deliberately does not own target artifact route ids, direct C
// artifact kinds, runtime ABI identities, or runtime glue roles.
struct ScalarBinaryMicrokernelRecord {
  const rvv::RVVBinaryFamilyRecord *rvvFamily = nullptr;
  std::string microkernelOpName;
  std::string operationNoun;
  std::string functionStem;
  std::string headerGuardStem;
};

// RVV+scalar dispatch family metadata. Dispatch source, ABI, component group,
// and external ABI identity are derived from selected component plans before
// this metadata may be used for dispatch checks.
struct RVVScalarDispatchFamilyRecord {
  const rvv::RVVBinaryFamilyRecord *rvvFamily = nullptr;
  std::string diagnosticName;
  std::string operationNoun;
  std::string functionStem;
  std::string headerGuardStem;
  std::string selfCheckSuccessMarker;
};

// Bridge registration metadata for the bounded finite RVV+scalar family set.
// This is not a legacy direct C path or selected family authority.
struct RVVScalarBinaryFamilyRecord {
  const rvv::RVVBinaryFamilyRecord *rvvFamily = nullptr;
  std::string familyID;
  ScalarBinaryMicrokernelRecord scalar;
  RVVScalarDispatchFamilyRecord dispatch;
};

struct ScalarBinarySelectedPlanMetadataRecord {
  llvm::StringRef name;
  llvm::StringRef value;
  llvm::StringRef role;
  llvm::StringRef note;
};

inline llvm::StringRef getScalarSelectedBinaryDTypeMetadataName() {
  return "tcrv_scalar.selected_binary_dtype";
}

inline llvm::StringRef getScalarSelectedBinaryFamilyMetadataName() {
  return "tcrv_scalar.selected_binary_family";
}

inline llvm::StringRef getScalarSelectedBinaryOperatorMetadataName() {
  return "tcrv_scalar.selected_binary_operator";
}

inline llvm::StringRef getScalarRuntimeElementCountCNameMetadataName() {
  return "tcrv_scalar.runtime_element_count_c_name";
}

inline llvm::StringRef getScalarEmitCSourceOpMetadataName() {
  return "tcrv_scalar.emitc_source_op";
}

inline llvm::StringRef getScalarEmitCLowerableOpInterfaceMetadataName() {
  return "tcrv_scalar.emitc_lowerable_op_interface";
}

inline llvm::StringRef getScalarTypedBinarySourceMetadataRole() {
  return "typed-scalar-binary-source";
}

inline llvm::StringRef getScalarTypedBinarySourceMetadataNote() {
  return "typed scalar family op metadata selected by the scalar plugin for "
         "the common EmitC route; not metadata-owned computation, runtime "
         "correctness evidence, or performance evidence";
}

inline llvm::StringRef getScalarEmitCSourceOpMetadataRole() {
  return "typed-scalar-emitc-source-op";
}

inline llvm::StringRef getScalarEmitCSourceOpMetadataNote() {
  return "typed scalar microkernel op used as the source operation for the "
         "common EmitC lowerable route; not a legacy route label";
}

inline llvm::StringRef getScalarEmitCLowerableOpInterfaceMetadataNote() {
  return "generated scalar op interface queried before building the common "
         "EmitC lowerable route; not metadata-selected computation";
}

inline llvm::StringRef getScalarRuntimeControlNameMetadataRole() {
  return "scalar-runtime-control-name-boundary";
}

inline llvm::StringRef getScalarRuntimeControlNameMetadataNote() {
  return "runtime ABI/control C name resolved from tcrv.exec runtime boundary "
         "for the scalar runtime handoff; not artifact-local component "
         "capacity, a tensor shape, or hardware evidence";
}

inline std::string makeScalarRegistrationMicrokernelOpName(
    const rvv::RVVBinaryFamilyRecord &family) {
  return (llvm::Twine("tcrv_scalar.") + family.functionStem +
          "_microkernel")
      .str();
}

inline std::string makeRegistrationOperationNoun(
    const rvv::RVVBinaryFamilyRecord &family) {
  return (llvm::Twine(family.dtypeID) + " vector-" +
          family.arithmeticVerb)
      .str();
}

inline std::string makeDispatchRegistrationSelfCheckSuccessMarker(
    const rvv::RVVBinaryFamilyRecord &family) {
  return (llvm::Twine("tcrv_rvv_scalar_") + family.functionStem +
          "_dispatch_self_check_ok")
      .str();
}

inline RVVScalarBinaryFamilyRecord makeFamilyRegistrationRecord(
    const rvv::RVVBinaryFamilyRecord &family) {
  RVVScalarBinaryFamilyRecord descriptor;
  descriptor.rvvFamily = &family;
  descriptor.familyID = family.familyID.str();

  descriptor.scalar.rvvFamily = &family;
  descriptor.scalar.microkernelOpName = makeScalarRegistrationMicrokernelOpName(family);
  descriptor.scalar.operationNoun = makeRegistrationOperationNoun(family);
  descriptor.scalar.functionStem = family.functionStem.str();
  descriptor.scalar.headerGuardStem = family.headerGuardStem.str();

  descriptor.dispatch.rvvFamily = &family;
  descriptor.dispatch.diagnosticName = family.familyID.str();
  descriptor.dispatch.operationNoun = makeRegistrationOperationNoun(family);
  descriptor.dispatch.functionStem = family.functionStem.str();
  descriptor.dispatch.headerGuardStem = family.headerGuardStem.str();
  descriptor.dispatch.selfCheckSuccessMarker =
      makeDispatchRegistrationSelfCheckSuccessMarker(family);
  return descriptor;
}

inline void appendScalarBinarySelectedTypedSourceMetadata(
    const RVVScalarBinaryFamilyRecord &family,
    llvm::StringRef runtimeElementCountCName,
    llvm::SmallVectorImpl<ScalarBinarySelectedPlanMetadataRecord> &out) {
  llvm::StringRef typedRole = getScalarTypedBinarySourceMetadataRole();
  llvm::StringRef typedNote = getScalarTypedBinarySourceMetadataNote();
  out.push_back({getScalarSelectedBinaryDTypeMetadataName(),
                 family.rvvFamily->dtypeID, typedRole, typedNote});
  out.push_back({getScalarSelectedBinaryFamilyMetadataName(),
                 family.familyID, typedRole, typedNote});
  out.push_back({getScalarSelectedBinaryOperatorMetadataName(),
                 family.rvvFamily->arithmeticVerb, typedRole, typedNote});
  out.push_back({getScalarEmitCSourceOpMetadataName(),
                 family.scalar.microkernelOpName,
                 getScalarEmitCSourceOpMetadataRole(),
                 getScalarEmitCSourceOpMetadataNote()});
  out.push_back({getScalarEmitCLowerableOpInterfaceMetadataName(),
                 "TCRVEmitCLowerableOpInterface",
                 getScalarEmitCSourceOpMetadataRole(),
                 getScalarEmitCLowerableOpInterfaceMetadataNote()});
  out.push_back({getScalarRuntimeElementCountCNameMetadataName(),
                 runtimeElementCountCName,
                 getScalarRuntimeControlNameMetadataRole(),
                 getScalarRuntimeControlNameMetadataNote()});
}

inline void appendScalarI32VAddSelectedTypedSourceMetadata(
    const RVVScalarBinaryFamilyRecord &family,
    llvm::StringRef runtimeElementCountCName,
    llvm::SmallVectorImpl<ScalarBinarySelectedPlanMetadataRecord> &out) {
  appendScalarBinarySelectedTypedSourceMetadata(family,
                                               runtimeElementCountCName, out);
}

inline const RVVScalarBinaryFamilyRecord &getI32VAddFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyRecord descriptor =
      makeFamilyRegistrationRecord(rvv::getI32VAddFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyRecord &getI32VSubFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyRecord descriptor =
      makeFamilyRegistrationRecord(rvv::getI32VSubFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyRecord &getI32VMulFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyRecord descriptor =
      makeFamilyRegistrationRecord(rvv::getI32VMulFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyRecord &getI64VAddFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyRecord descriptor =
      makeFamilyRegistrationRecord(rvv::getI64VAddFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyRecord &getI64VSubFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyRecord descriptor =
      makeFamilyRegistrationRecord(rvv::getI64VSubFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyRecord &getI64VMulFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyRecord descriptor =
      makeFamilyRegistrationRecord(rvv::getI64VMulFamilyRegistrationRecord());
  return descriptor;
}

inline llvm::ArrayRef<const RVVScalarBinaryFamilyRecord *>
getRVVScalarBinaryRegistrationRecords() {
  static const RVVScalarBinaryFamilyRecord *families[] = {
      &getI32VAddFamilyRegistrationRecord(), &getI32VSubFamilyRegistrationRecord(),
      &getI32VMulFamilyRegistrationRecord(), &getI64VAddFamilyRegistrationRecord(),
      &getI64VSubFamilyRegistrationRecord(), &getI64VMulFamilyRegistrationRecord()};
  return llvm::ArrayRef(families);
}

inline const RVVScalarBinaryFamilyRecord *
lookupRVVScalarBinaryRegistrationByID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const RVVScalarBinaryFamilyRecord *descriptor :
       getRVVScalarBinaryRegistrationRecords())
    if (descriptor->familyID == familyID)
      return descriptor;
  return nullptr;
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVScalarBinaryCallableRuntimeABIParameters(
    const RVVScalarBinaryFamilyRecord &family,
    llvm::StringRef runtimeCountCName = "n") {
  return rvv::getRVVBinaryCallableRuntimeABIParameters(*family.rvvFamily,
                                                       runtimeCountCName);
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVScalarBinaryCallableRuntimeABIRoleRequirements(
    const RVVScalarBinaryFamilyRecord &family) {
  return rvv::getRVVBinaryCallableRuntimeABIRoleRequirements(
      *family.rvvFamily);
}

inline llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3>
getRVVScalarBinaryBufferMemWindowSpecs(
    const RVVScalarBinaryFamilyRecord &family) {
  return rvv::getRVVBinaryBufferMemWindowSpecs(*family.rvvFamily);
}

inline llvm::SmallVector<support::RuntimeABIParamSpec, 1>
getRVVScalarBinaryRuntimeElementCountParamSpecs(
    const RVVScalarBinaryFamilyRecord &family,
    llvm::StringRef cName = "n") {
  return rvv::getRVVBinaryRuntimeElementCountParamSpecs(*family.rvvFamily,
                                                        cName);
}

inline support::RuntimeABIParameter
makeRVVScalarDispatchAvailabilityGuardParameter(
    llvm::StringRef cName = "rvv_available") {
  return support::makeTargetExportABIParameter(
      cName, "int",
      support::RuntimeABIParameterRole::DispatchAvailabilityGuard);
}

inline llvm::SmallVector<support::RuntimeABIParamSpec, 2>
getRVVScalarDispatchRuntimeParamSpecs(
    const RVVScalarBinaryFamilyRecord &family,
    llvm::StringRef runtimeCountCName = "n",
    llvm::StringRef guardCName = "rvv_available") {
  llvm::SmallVector<support::RuntimeABIParamSpec, 2> specs;
  auto countSpecs =
      getRVVScalarBinaryRuntimeElementCountParamSpecs(family,
                                                      runtimeCountCName);
  specs.append(countSpecs.begin(), countSpecs.end());
  specs.push_back(support::getDispatchAvailabilityGuardParamSpec(guardCName));
  return specs;
}

inline llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVScalarDispatchRuntimeABIParameters(
    const RVVScalarBinaryFamilyRecord &family,
    llvm::StringRef runtimeCountCName = "n",
    llvm::StringRef guardCName = "rvv_available") {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  auto callable = getRVVScalarBinaryCallableRuntimeABIParameters(
      family, runtimeCountCName);
  parameters.append(callable.begin(), callable.end());
  parameters.push_back(
      makeRVVScalarDispatchAvailabilityGuardParameter(guardCName));
  return parameters;
}

} // namespace tianchenrv::target::rvv_scalar

#endif // TIANCHENRV_TARGET_RVVSCALARBINARYFAMILY_H
