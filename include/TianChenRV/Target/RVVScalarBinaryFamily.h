#ifndef TIANCHENRV_TARGET_RVVSCALARBINARYFAMILY_H
#define TIANCHENRV_TARGET_RVVSCALARBINARYFAMILY_H

#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace tianchenrv::target::rvv_scalar {

// Scalar route registration/compatibility metadata derived from the finite RVV
// family table. Production scalar artifacts must establish typed selected-plan
// authority before consulting this metadata.
struct ScalarBinaryMicrokernelDescriptor {
  const rvv::RVVBinaryFamilyDescriptor *rvvFamily = nullptr;
  std::string microkernelOpName;
  std::string operationNoun;
  std::string functionStem;
  std::string headerGuardStem;
  std::string descriptor;
  std::string emissionKind;
  std::string routeID;
  std::string headerRouteID;
  std::string objectRouteID;
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  std::string cOperator;
};

// RVV+scalar dispatch route registration/compatibility metadata. Dispatch
// source, ABI, component group, and external ABI identity are derived from
// selected component plans before this metadata may be used for route checks.
struct DispatchBinaryFamilyDescriptor {
  const rvv::RVVBinaryFamilyDescriptor *rvvFamily = nullptr;
  std::string diagnosticName;
  std::string operationNoun;
  std::string functionStem;
  std::string headerGuardStem;
  std::string cOperator;
  std::string selfCheckSuccessMarker;
  std::string rvvRouteID;
  std::string rvvEmissionKind;
  std::string rvvRuntimeABI;
  std::string rvvRuntimeABIKind;
  std::string rvvRuntimeABIName;
  std::string rvvRuntimeGlueRole;
  std::string scalarRouteID;
  std::string scalarEmissionKind;
  std::string scalarRuntimeABI;
  std::string scalarRuntimeABIKind;
  std::string scalarRuntimeABIName;
  std::string scalarRuntimeGlueRole;
  std::string dispatchSourceRouteID;
  std::string dispatchHeaderRouteID;
  std::string dispatchObjectRouteID;
  std::string dispatchSelfCheckSourceRouteID;
  std::string dispatchSelfCheckObjectRouteID;
  std::string dispatchRuntimeABIKind;
  std::string dispatchRuntimeABIName;
  std::string dispatchExternalABIComponentGroup;
};

// Bridge registration metadata for the bounded finite RVV+scalar family set.
// This is not a descriptor-to-C path or selected family authority.
struct RVVScalarBinaryFamilyDescriptor {
  const rvv::RVVBinaryFamilyDescriptor *rvvFamily = nullptr;
  std::string familyID;
  std::string frontendLowering;
  std::string loweringDescriptor;
  std::string descriptorNoun;
  ScalarBinaryMicrokernelDescriptor scalar;
  DispatchBinaryFamilyDescriptor dispatch;
};

struct ScalarBinarySelectedPlanMetadataDescriptor {
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

inline llvm::StringRef getScalarSelectedLoweringDescriptorMetadataName() {
  return "tcrv_scalar.selected_lowering_descriptor";
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

inline llvm::StringRef getScalarLegacyDescriptorMirrorMetadataRole() {
  return "legacy-scalar-binary-descriptor-mirror";
}

inline llvm::StringRef getScalarTypedBinarySourceMetadataRole() {
  return "typed-scalar-binary-source";
}

inline llvm::StringRef getScalarLegacyDescriptorMirrorMetadataNote() {
  return "legacy finite scalar descriptor mirror metadata checked after typed "
         "selected-plan authority is established; not compute, ABI, source, "
         "runtime trip-count, hardware execution, or performance authority";
}

inline llvm::StringRef getScalarTypedBinarySourceMetadataNote() {
  return "typed scalar family op metadata selected by the scalar plugin for "
         "the common EmitC route; not descriptor-owned computation, runtime "
         "correctness evidence, or performance evidence";
}

inline llvm::StringRef getScalarEmitCSourceOpMetadataRole() {
  return "typed-scalar-emitc-source-op";
}

inline llvm::StringRef getScalarEmitCSourceOpMetadataNote() {
  return "typed scalar microkernel op used as the source operation for the "
         "common EmitC lowerable route; not a lowering descriptor";
}

inline llvm::StringRef getScalarEmitCLowerableOpInterfaceMetadataNote() {
  return "generated scalar op interface queried before building the common "
         "EmitC lowerable route; not descriptor-selected computation";
}

inline llvm::StringRef getScalarRuntimeControlNameMetadataRole() {
  return "scalar-runtime-control-name-boundary";
}

inline llvm::StringRef getScalarRuntimeControlNameMetadataNote() {
  return "runtime ABI/control C name resolved from tcrv.exec runtime boundary "
         "for the scalar callable source route; not descriptor-local "
         "element_count, a tensor shape, or hardware evidence";
}

inline std::string makeScalarRegistrationMicrokernelOpName(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("tcrv_scalar.") + family.functionStem +
          "_microkernel")
      .str();
}

inline std::string makeRegistrationOperationNoun(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine(family.dtypeID) + " vector-" +
          family.arithmeticVerb)
      .str();
}

inline std::string makeScalarRegistrationEmissionKind(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("scalar-explicit-") + family.familyID +
          "-microkernel-c-source")
      .str();
}

inline std::string makeScalarCompatibilityRouteID(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  if (family.familyID == "i32-vadd")
    return "tcrv-export-scalar-microkernel-c";
  return (llvm::Twine("tcrv-export-scalar-") + family.familyID +
          "-microkernel-c")
      .str();
}

inline std::string makeScalarCompatibilityHeaderRouteID(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  if (family.familyID == "i32-vadd")
    return "tcrv-export-scalar-microkernel-header";
  return (llvm::Twine("tcrv-export-scalar-") + family.familyID +
          "-microkernel-header")
      .str();
}

inline std::string makeScalarCompatibilityObjectRouteID(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  if (family.familyID == "i32-vadd")
    return "tcrv-export-scalar-microkernel-object";
  return (llvm::Twine("tcrv-export-scalar-") + family.familyID +
          "-microkernel-object")
      .str();
}

inline std::string makeScalarCompatibilityRuntimeABI(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("scalar-") + family.familyID +
          "-runtime-callable-c-abi.v1")
      .str();
}

inline std::string makeScalarCompatibilityRuntimeABIName(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("scalar-") + family.familyID +
          "-runtime-callable-c-function.v1")
      .str();
}

inline std::string makeScalarCompatibilityRuntimeGlueRole(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("runtime-callable-") + family.familyID +
          "-fallback-function")
      .str();
}

inline std::string makeDispatchRegistrationRouteID(
    const rvv::RVVBinaryFamilyDescriptor &family, llvm::StringRef suffix) {
  return (llvm::Twine("tcrv-export-rvv-scalar-") + family.familyID +
          "-dispatch-" + suffix)
      .str();
}

inline std::string makeDispatchRegistrationRuntimeABIName(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("rvv-scalar-") + family.familyID +
          "-dispatch-runtime-callable-c-function.v1")
      .str();
}

inline std::string makeDispatchRegistrationExternalABIComponentGroup(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("rvv-scalar-") + family.familyID +
          "-dispatch-external-abi.v1")
      .str();
}

inline std::string makeDispatchRegistrationSelfCheckSuccessMarker(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("tcrv_rvv_scalar_") + family.functionStem +
          "_dispatch_self_check_ok")
      .str();
}

inline RVVScalarBinaryFamilyDescriptor makeFamilyRegistrationRecord(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  RVVScalarBinaryFamilyDescriptor descriptor;
  descriptor.rvvFamily = &family;
  descriptor.familyID = family.familyID.str();
  descriptor.frontendLowering = family.frontendLowering.str();
  descriptor.loweringDescriptor = family.loweringDescriptor.str();
  descriptor.descriptorNoun =
      (llvm::Twine("finite scalar ") + family.familyID +
       " lowering descriptor")
          .str();

  descriptor.scalar.rvvFamily = &family;
  descriptor.scalar.microkernelOpName = makeScalarRegistrationMicrokernelOpName(family);
  descriptor.scalar.operationNoun = makeRegistrationOperationNoun(family);
  descriptor.scalar.functionStem = family.functionStem.str();
  descriptor.scalar.headerGuardStem = family.headerGuardStem.str();
  descriptor.scalar.descriptor = family.loweringDescriptor.str();
  descriptor.scalar.emissionKind = makeScalarRegistrationEmissionKind(family);
  descriptor.scalar.routeID = makeScalarCompatibilityRouteID(family);
  descriptor.scalar.headerRouteID = makeScalarCompatibilityHeaderRouteID(family);
  descriptor.scalar.objectRouteID = makeScalarCompatibilityObjectRouteID(family);
  descriptor.scalar.runtimeABI = makeScalarCompatibilityRuntimeABI(family);
  descriptor.scalar.runtimeABIKind = "scalar-runtime-callable-c-abi";
  descriptor.scalar.runtimeABIName = makeScalarCompatibilityRuntimeABIName(family);
  descriptor.scalar.runtimeGlueRole = makeScalarCompatibilityRuntimeGlueRole(family);
  descriptor.scalar.cOperator = family.cOperator.str();

  descriptor.dispatch.rvvFamily = &family;
  descriptor.dispatch.diagnosticName = family.familyID.str();
  descriptor.dispatch.operationNoun = makeRegistrationOperationNoun(family);
  descriptor.dispatch.functionStem = family.functionStem.str();
  descriptor.dispatch.headerGuardStem = family.headerGuardStem.str();
  descriptor.dispatch.cOperator = family.cOperator.str();
  descriptor.dispatch.selfCheckSuccessMarker =
      makeDispatchRegistrationSelfCheckSuccessMarker(family);
  descriptor.dispatch.rvvRouteID = family.routeID.str();
  descriptor.dispatch.rvvEmissionKind = family.emissionKind.str();
  descriptor.dispatch.rvvRuntimeABI = family.runtimeABI.str();
  descriptor.dispatch.rvvRuntimeABIKind = family.runtimeABIKind.str();
  descriptor.dispatch.rvvRuntimeABIName = family.runtimeABIName.str();
  descriptor.dispatch.rvvRuntimeGlueRole = family.runtimeGlueRole.str();
  descriptor.dispatch.scalarRouteID = descriptor.scalar.routeID;
  descriptor.dispatch.scalarEmissionKind = descriptor.scalar.emissionKind;
  descriptor.dispatch.scalarRuntimeABI = descriptor.scalar.runtimeABI;
  descriptor.dispatch.scalarRuntimeABIKind = descriptor.scalar.runtimeABIKind;
  descriptor.dispatch.scalarRuntimeABIName = descriptor.scalar.runtimeABIName;
  descriptor.dispatch.scalarRuntimeGlueRole =
      descriptor.scalar.runtimeGlueRole;
  descriptor.dispatch.dispatchSourceRouteID =
      makeDispatchRegistrationRouteID(family, "c");
  descriptor.dispatch.dispatchHeaderRouteID =
      makeDispatchRegistrationRouteID(family, "header");
  descriptor.dispatch.dispatchObjectRouteID =
      makeDispatchRegistrationRouteID(family, "object");
  descriptor.dispatch.dispatchSelfCheckSourceRouteID =
      makeDispatchRegistrationRouteID(family, "self-check-c");
  descriptor.dispatch.dispatchSelfCheckObjectRouteID =
      makeDispatchRegistrationRouteID(family, "self-check-object");
  descriptor.dispatch.dispatchRuntimeABIKind =
      "rvv-scalar-dispatch-runtime-callable-c-abi";
  descriptor.dispatch.dispatchRuntimeABIName =
      makeDispatchRegistrationRuntimeABIName(family);
  descriptor.dispatch.dispatchExternalABIComponentGroup =
      makeDispatchRegistrationExternalABIComponentGroup(family);
  return descriptor;
}

inline void appendScalarBinaryLegacyDescriptorMirrorMetadata(
    const RVVScalarBinaryFamilyDescriptor &family,
    llvm::StringRef runtimeElementCountCName,
    llvm::SmallVectorImpl<ScalarBinarySelectedPlanMetadataDescriptor> &out) {
  llvm::StringRef descriptorRole =
      getScalarLegacyDescriptorMirrorMetadataRole();
  llvm::StringRef descriptorNote =
      getScalarLegacyDescriptorMirrorMetadataNote();
  out.push_back({getScalarSelectedBinaryDTypeMetadataName(),
                 family.rvvFamily->dtypeID, descriptorRole, descriptorNote});
  out.push_back({getScalarSelectedBinaryFamilyMetadataName(),
                 family.familyID, descriptorRole, descriptorNote});
  out.push_back({getScalarSelectedBinaryOperatorMetadataName(),
                 family.rvvFamily->arithmeticVerb, descriptorRole,
                 descriptorNote});
  out.push_back({getScalarSelectedLoweringDescriptorMetadataName(),
                 family.loweringDescriptor, descriptorRole, descriptorNote});
  out.push_back({getScalarRuntimeElementCountCNameMetadataName(),
                 runtimeElementCountCName,
                 getScalarRuntimeControlNameMetadataRole(),
                 getScalarRuntimeControlNameMetadataNote()});
}

inline void appendScalarBinarySelectedTypedSourceMetadata(
    const RVVScalarBinaryFamilyDescriptor &family,
    llvm::StringRef runtimeElementCountCName,
    llvm::SmallVectorImpl<ScalarBinarySelectedPlanMetadataDescriptor> &out) {
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
    const RVVScalarBinaryFamilyDescriptor &family,
    llvm::StringRef runtimeElementCountCName,
    llvm::SmallVectorImpl<ScalarBinarySelectedPlanMetadataDescriptor> &out) {
  appendScalarBinarySelectedTypedSourceMetadata(family,
                                               runtimeElementCountCName, out);
}

inline const RVVScalarBinaryFamilyDescriptor &getI32VAddFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyRegistrationRecord(rvv::getI32VAddFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI32VSubFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyRegistrationRecord(rvv::getI32VSubFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI32VMulFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyRegistrationRecord(rvv::getI32VMulFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI64VAddFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyRegistrationRecord(rvv::getI64VAddFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI64VSubFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyRegistrationRecord(rvv::getI64VSubFamilyRegistrationRecord());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI64VMulFamilyRegistrationRecord() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyRegistrationRecord(rvv::getI64VMulFamilyRegistrationRecord());
  return descriptor;
}

inline llvm::ArrayRef<const RVVScalarBinaryFamilyDescriptor *>
getRVVScalarBinaryRegistrationRecords() {
  static const RVVScalarBinaryFamilyDescriptor *families[] = {
      &getI32VAddFamilyRegistrationRecord(), &getI32VSubFamilyRegistrationRecord(),
      &getI32VMulFamilyRegistrationRecord(), &getI64VAddFamilyRegistrationRecord(),
      &getI64VSubFamilyRegistrationRecord(), &getI64VMulFamilyRegistrationRecord()};
  return llvm::ArrayRef(families);
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryRegistrationByID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryRegistrationRecords())
    if (descriptor->familyID == familyID)
      return descriptor;
  return nullptr;
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryRegistrationByFrontendLowering(
    llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryRegistrationRecords())
    if (descriptor->frontendLowering == frontendLowering)
      return descriptor;
  return nullptr;
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryRegistrationByLegacyLoweringDescriptor(
    llvm::StringRef loweringDescriptor) {
  loweringDescriptor = loweringDescriptor.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryRegistrationRecords())
    if (descriptor->loweringDescriptor == loweringDescriptor)
      return descriptor;
  return nullptr;
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryRegistrationByScalarRouteID(llvm::StringRef routeID) {
  routeID = routeID.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryRegistrationRecords())
    if (descriptor->scalar.routeID == routeID)
      return descriptor;
  return nullptr;
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVScalarBinaryCallableRuntimeABIParameters(
    const RVVScalarBinaryFamilyDescriptor &family,
    llvm::StringRef runtimeCountCName = "n") {
  return rvv::getRVVBinaryCallableRuntimeABIParameters(*family.rvvFamily,
                                                       runtimeCountCName);
}

inline llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVScalarBinaryCallableRuntimeABIRoleRequirements(
    const RVVScalarBinaryFamilyDescriptor &family) {
  return rvv::getRVVBinaryCallableRuntimeABIRoleRequirements(
      *family.rvvFamily);
}

inline llvm::SmallVector<support::RuntimeABIMemWindowSpec, 3>
getRVVScalarBinaryBufferMemWindowSpecs(
    const RVVScalarBinaryFamilyDescriptor &family) {
  return rvv::getRVVBinaryBufferMemWindowSpecs(*family.rvvFamily);
}

inline llvm::SmallVector<support::RuntimeABIParamSpec, 1>
getRVVScalarBinaryRuntimeElementCountParamSpecs(
    const RVVScalarBinaryFamilyDescriptor &family,
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
    const RVVScalarBinaryFamilyDescriptor &family,
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
    const RVVScalarBinaryFamilyDescriptor &family,
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
