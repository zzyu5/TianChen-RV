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

struct RVVScalarBinaryFamilyDescriptor {
  const rvv::RVVBinaryFamilyDescriptor *rvvFamily = nullptr;
  std::string familyID;
  std::string frontendLowering;
  std::string loweringDescriptor;
  std::string descriptorNoun;
  ScalarBinaryMicrokernelDescriptor scalar;
  DispatchBinaryFamilyDescriptor dispatch;
};

inline std::string hyphenatedFamilyID(const rvv::RVVBinaryFamilyDescriptor &family) {
  return family.familyID.str();
}

inline std::string underscoredFunctionStem(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return family.functionStem.str();
}

inline std::string makeScalarMicrokernelOpName(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("tcrv_scalar.") + family.functionStem +
          "_microkernel")
      .str();
}

inline std::string makeOperationNoun(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine(family.dtypeID) + " vector-" +
          family.arithmeticVerb)
      .str();
}

inline std::string makeScalarEmissionKind(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("scalar-explicit-") + family.familyID +
          "-microkernel-c-source")
      .str();
}

inline std::string makeScalarRouteID(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  if (family.familyID == "i32-vadd")
    return "tcrv-export-scalar-microkernel-c";
  return (llvm::Twine("tcrv-export-scalar-") + family.familyID +
          "-microkernel-c")
      .str();
}

inline std::string makeScalarRuntimeABI(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("scalar-") + family.familyID +
          "-runtime-callable-c-abi.v1")
      .str();
}

inline std::string makeScalarRuntimeABIName(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("scalar-") + family.familyID +
          "-runtime-callable-c-function.v1")
      .str();
}

inline std::string makeScalarRuntimeGlueRole(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("runtime-callable-") + family.familyID +
          "-fallback-function")
      .str();
}

inline std::string makeDispatchRouteID(
    const rvv::RVVBinaryFamilyDescriptor &family, llvm::StringRef suffix) {
  return (llvm::Twine("tcrv-export-rvv-scalar-") + family.familyID +
          "-dispatch-" + suffix)
      .str();
}

inline std::string makeDispatchRuntimeABIName(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("rvv-scalar-") + family.familyID +
          "-dispatch-runtime-callable-c-function.v1")
      .str();
}

inline std::string makeDispatchExternalABIComponentGroup(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("rvv-scalar-") + family.familyID +
          "-dispatch-external-abi.v1")
      .str();
}

inline std::string makeDispatchSelfCheckSuccessMarker(
    const rvv::RVVBinaryFamilyDescriptor &family) {
  return (llvm::Twine("tcrv_rvv_scalar_") + family.functionStem +
          "_dispatch_self_check_ok")
      .str();
}

inline RVVScalarBinaryFamilyDescriptor makeFamilyDescriptor(
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
  descriptor.scalar.microkernelOpName = makeScalarMicrokernelOpName(family);
  descriptor.scalar.operationNoun = makeOperationNoun(family);
  descriptor.scalar.functionStem = family.functionStem.str();
  descriptor.scalar.headerGuardStem = family.headerGuardStem.str();
  descriptor.scalar.descriptor = family.loweringDescriptor.str();
  descriptor.scalar.emissionKind = makeScalarEmissionKind(family);
  descriptor.scalar.routeID = makeScalarRouteID(family);
  descriptor.scalar.headerRouteID = "tcrv-export-scalar-microkernel-header";
  descriptor.scalar.objectRouteID = "tcrv-export-scalar-microkernel-object";
  descriptor.scalar.runtimeABI = makeScalarRuntimeABI(family);
  descriptor.scalar.runtimeABIKind = "scalar-runtime-callable-c-abi";
  descriptor.scalar.runtimeABIName = makeScalarRuntimeABIName(family);
  descriptor.scalar.runtimeGlueRole = makeScalarRuntimeGlueRole(family);
  descriptor.scalar.cOperator = family.cOperator.str();

  descriptor.dispatch.rvvFamily = &family;
  descriptor.dispatch.diagnosticName = family.familyID.str();
  descriptor.dispatch.operationNoun = makeOperationNoun(family);
  descriptor.dispatch.functionStem = family.functionStem.str();
  descriptor.dispatch.headerGuardStem = family.headerGuardStem.str();
  descriptor.dispatch.cOperator = family.cOperator.str();
  descriptor.dispatch.selfCheckSuccessMarker =
      makeDispatchSelfCheckSuccessMarker(family);
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
      makeDispatchRouteID(family, "c");
  descriptor.dispatch.dispatchHeaderRouteID =
      makeDispatchRouteID(family, "header");
  descriptor.dispatch.dispatchObjectRouteID =
      makeDispatchRouteID(family, "object");
  descriptor.dispatch.dispatchSelfCheckSourceRouteID =
      makeDispatchRouteID(family, "self-check-c");
  descriptor.dispatch.dispatchSelfCheckObjectRouteID =
      makeDispatchRouteID(family, "self-check-object");
  descriptor.dispatch.dispatchRuntimeABIKind =
      "rvv-scalar-dispatch-runtime-callable-c-abi";
  descriptor.dispatch.dispatchRuntimeABIName =
      makeDispatchRuntimeABIName(family);
  descriptor.dispatch.dispatchExternalABIComponentGroup =
      makeDispatchExternalABIComponentGroup(family);
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI32VAddFamilyDescriptor() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyDescriptor(rvv::getI32VAddFamilyDescriptor());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI32VSubFamilyDescriptor() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyDescriptor(rvv::getI32VSubFamilyDescriptor());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI32VMulFamilyDescriptor() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyDescriptor(rvv::getI32VMulFamilyDescriptor());
  return descriptor;
}

inline const RVVScalarBinaryFamilyDescriptor &getI64VAddFamilyDescriptor() {
  static const RVVScalarBinaryFamilyDescriptor descriptor =
      makeFamilyDescriptor(rvv::getI64VAddFamilyDescriptor());
  return descriptor;
}

inline llvm::ArrayRef<const RVVScalarBinaryFamilyDescriptor *>
getRVVScalarBinaryFamilyDescriptors() {
  static const RVVScalarBinaryFamilyDescriptor *families[] = {
      &getI32VAddFamilyDescriptor(), &getI32VSubFamilyDescriptor(),
      &getI32VMulFamilyDescriptor(), &getI64VAddFamilyDescriptor()};
  return llvm::ArrayRef(families);
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryFamilyByID(llvm::StringRef familyID) {
  familyID = familyID.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryFamilyDescriptors())
    if (descriptor->familyID == familyID)
      return descriptor;
  return nullptr;
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryFamilyByFrontendLowering(
    llvm::StringRef frontendLowering) {
  frontendLowering = frontendLowering.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryFamilyDescriptors())
    if (descriptor->frontendLowering == frontendLowering)
      return descriptor;
  return nullptr;
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryFamilyByLoweringDescriptor(
    llvm::StringRef loweringDescriptor) {
  loweringDescriptor = loweringDescriptor.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryFamilyDescriptors())
    if (descriptor->loweringDescriptor == loweringDescriptor)
      return descriptor;
  return nullptr;
}

inline const RVVScalarBinaryFamilyDescriptor *
lookupRVVScalarBinaryFamilyByScalarRouteID(llvm::StringRef routeID) {
  routeID = routeID.trim();
  for (const RVVScalarBinaryFamilyDescriptor *descriptor :
       getRVVScalarBinaryFamilyDescriptors())
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
