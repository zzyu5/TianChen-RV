#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace tianchenrv::target::rvv {
namespace {

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

llvm::Error makeRVVTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV materialized EmitC target artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

constexpr llvm::StringLiteral kRVVMaskTailPolicyRouteFamilyPlanID(
    "rvv-mask-tail-policy-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectMaskTailPolicyOwner(
    "computed-mask select mask/tail policy");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskTailPolicyOwner(
    "computed-mask memory mask/tail policy");

const plugin::rvv::RVVConstructionManifest &getRVVManifest() {
  return plugin::rvv::getRVVConstructionManifest();
}

const plugin::rvv::RVVSelectedBodyTargetArtifactMapping &
getRVVTargetMapping() {
  return plugin::rvv::getRVVSelectedBodyTargetArtifactMapping();
}

llvm::Error requireCandidateField(llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine("candidate ") + fieldName +
                                 " must be '" + expected + "' but was '" +
                                 actual + "'");
}

llvm::StringRef lookupCandidateMetadataValue(
    const TargetArtifactCandidate &candidate, llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

bool isRVVSegment2RouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2DeinterleaveUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2InterleaveUnitLoad:
    return true;
  default:
    return false;
  }
}

bool isRVVComputedMaskSegment2RouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2LoadUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2StoreUnitLoad ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2UpdateUnitLoad;
}

bool isRVVPlainSegment2RouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 Segment2DeinterleaveUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 Segment2InterleaveUnitLoad;
}

bool isRVVMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVScalarBroadcastMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
}

bool isRVVComputedMaskMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          RuntimeScalarComputedMaskedMAccAdd;
}

bool isRVVStandaloneReductionAccumulationRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStandaloneReduceAdd ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStandaloneReduceMin ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStandaloneReduceMax ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceAdd ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceMin ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceMax;
}

bool isRVVComputedMaskStandaloneReductionAccumulationRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStandaloneReduceAdd ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStandaloneReduceMin ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskStandaloneReduceMax ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceAdd ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceMin ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceMax;
}

bool isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceAdd ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceMin ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 RuntimeScalarComputedMaskStandaloneReduceMax;
}

bool isRVVCompareSelectMaskProducerRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return true;
  default:
    return false;
  }
}

bool isRVVPlainCompareSelectMaskRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect;
}

bool isRVVComputedMaskSelectRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return true;
  default:
    return false;
  }
}

bool isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return true;
  default:
    return false;
  }
}

bool isRVVCompareSelectMaskRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVCompareSelectMaskProducerRouteFamilyOperation(operation) ||
         isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(operation);
}

bool isRVVConversionDtypePolicyScalarBroadcastRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMul:
    return true;
  default:
    return false;
  }
}

bool isRVVConversionDtypePolicyWideningRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI32ToI64:
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI16ToI32:
    return true;
  default:
    return false;
  }
}

bool isRVVConversionDtypePolicyRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVConversionDtypePolicyScalarBroadcastRouteFamilyOperation(
             operation) ||
         isRVVConversionDtypePolicyWideningRouteFamilyOperation(operation);
}

bool isRVVElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::Add:
  case plugin::rvv::RVVSelectedBodyOperationKind::Sub:
  case plugin::rvv::RVVSelectedBodyOperationKind::Mul:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedMul:
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedMul:
    return true;
  default:
    return false;
  }
}

bool isRVVStridedElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd;
}

bool isRVVElementwiseArithmeticRouteFamilyDescription(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVElementwiseArithmeticRouteFamilyOperation(description.operation))
    return false;
  if (isRVVStridedElementwiseArithmeticRouteFamilyOperation(
          description.operation))
    return description.memoryForm ==
           plugin::rvv::RVVSelectedBodyMemoryForm::StridedLoadStore;
  return description.memoryForm ==
         plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

bool isRVVRuntimeScalarSplatStoreRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::RuntimeI32SplatStore;
}

llvm::Error requireCandidateMetadataMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef expected, llvm::StringRef label) {
  llvm::StringRef actual = lookupCandidateMetadataValue(candidate, key);
  if (!expected.empty()) {
    if (actual.empty())
      return makeRVVTargetRouteError(llvm::Twine("candidate metadata must "
                                                 "carry ") +
                                     key + " provenance");
    if (actual != expected)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate ") + key + " provenance must mirror " +
          label + " '" + expected + "' but was '" + actual + "'");
  } else if (!actual.empty()) {
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must not carry ") + key +
        " mirrors for a selected typed RVV body route without " + label);
  }

  return llvm::Error::success();
}

bool routeHasHeader(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    llvm::StringRef expectedHeader) {
  for (const conversion::emitc::TCRVEmitCHeaderRequirement &header :
       route.getHeaders())
    if (header.header == expectedHeader)
      return true;
  return false;
}

bool routeHasTypeMapping(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    llvm::StringRef sourceType, llvm::StringRef cType) {
  for (const conversion::emitc::TCRVEmitCTypeMapping &mapping :
       route.getTypeMappings())
    if (mapping.sourceType == sourceType && mapping.cType == cType)
      return true;
  return false;
}

bool runtimeABIParameterEquals(const support::RuntimeABIParameter &lhs,
                               const support::RuntimeABIParameter &rhs) {
  return lhs.cName == rhs.cName && lhs.cType == rhs.cType &&
         lhs.role == rhs.role && lhs.ownership == rhs.ownership;
}

llvm::Error validateRVVCompareSelectMaskRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires at least one "
        "provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.maskTypeName.empty() ||
      description.maskCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived VL, vector, mask, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  if (!routeHasTypeMapping(route, description.maskTypeName,
                           description.maskCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.maskTypeName + "' -> '" + description.maskCType + "'");
  if (!description.indexVectorTypeName.empty()) {
    if (description.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires index vector "
          "C type facts when index vector type facts are present");
    if (!routeHasTypeMapping(route, description.indexVectorTypeName,
                             description.indexVectorCType))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route type mapping '") +
          description.indexVectorTypeName + "' -> '" +
          description.indexVectorCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVCompareSelectMaskRouteFamilyOperation(description.operation))
    return llvm::Error::success();

  if (description.comparePredicateKind.empty() ||
      description.compareIntrinsic.empty() || description.maskName.empty() ||
      description.maskRole.empty() || description.maskSource.empty() ||
      description.maskMemoryForm.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived compare predicate, compare intrinsic, mask result, "
        "mask role, mask source, and mask memory-form facts before artifact "
        "export");
  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation) &&
      description.selectLayout.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived selected-value/passthrough layout facts before "
        "artifact export");
  if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation) &&
      (description.inactiveLaneContract.empty() ||
       description.maskedPassthroughLayout.empty()))
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived inactive-lane and masked-passthrough layout facts "
        "before artifact export");
  if (isRVVComputedMaskSelectRouteFamilyOperation(description.operation) ||
      isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    llvm::StringRef expectedOwner =
        isRVVComputedMaskSelectRouteFamilyOperation(description.operation)
            ? llvm::StringRef(kRVVComputedMaskSelectMaskTailPolicyOwner)
            : llvm::StringRef(kRVVComputedMaskMemoryMaskTailPolicyOwner);
    if (description.maskTailPolicyRouteFamilyPlanID !=
            kRVVMaskTailPolicyRouteFamilyPlanID ||
        description.maskTailPolicyOwner != expectedOwner)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "provider-derived mask/tail policy owner '") +
          expectedOwner + "' and route-family plan '" +
          kRVVMaskTailPolicyRouteFamilyPlanID + "' before artifact export");
  } else if (!description.maskTailPolicyRouteFamilyPlanID.empty() ||
             !description.maskTailPolicyOwner.empty()) {
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer rejects mask/tail policy "
        "mirrors for routes that are not computed-mask select or "
        "compare-produced computed-mask memory");
  }

  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteTypeMappings(route, description))
    return error;
  return validateRVVCompareSelectMaskRouteABIMappings(route, description);
}

bool stepHasResult(const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
                   llvm::StringRef resultName, llvm::StringRef cType) {
  return step.result && step.result->name == resultName &&
         step.result->cType == cType;
}

bool routeStepSourceIsSelectedRVVBody(
    const conversion::emitc::TCRVEmitCCallOpaqueStep &step) {
  return step.sourceOp.opInterface ==
             plugin::rvv::getRVVEmitCLowerableOpInterfaceName() &&
         !step.sourceOp.opName.empty() && !step.sourceOp.role.empty();
}

bool routeStepsContainCallee(
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueStep> steps,
    llvm::StringRef callee) {
  if (callee.empty())
    return true;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : steps)
    if (step.callee == callee)
      return true;
  return false;
}

bool routeLoopContainsCallee(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    llvm::StringRef callee) {
  return routeStepsContainCallee(loop.bodySteps, callee);
}

llvm::StringRef getRVVElementwiseArithmeticSignedCType(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.elementTypeName == "i16")
    return "int16_t";
  if (description.elementTypeName == "i32")
    return "int32_t";
  if (description.elementTypeName == "i64")
    return "int64_t";
  return {};
}

const support::RuntimeABIParameter *findRuntimeElementCountABIParameter(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount)
      return &parameter;
  return nullptr;
}

llvm::Error validateRVVElementwiseArithmeticRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.elementTypeName.empty() || description.sew == 0 ||
      description.lmul.empty() || description.vlCType.empty() ||
      description.vectorTypeName.empty() || description.vectorCType.empty() ||
      description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived dtype, SEW, LMUL, VL, vector, and C type mapping "
        "facts before artifact export");

  if (getRVVElementwiseArithmeticSignedCType(description).empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires a "
        "provider-derived signed C type for the typed RVV element type");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("masked elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale mask "
        "type mapping facts for non-masked elementwise routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  llvm::StringRef signedCType =
      getRVVElementwiseArithmeticSignedCType(description);
  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");

    switch (expected.role) {
    case support::RuntimeABIParameterRole::LHSInputBuffer:
    case support::RuntimeABIParameterRole::RHSInputBuffer:
    case support::RuntimeABIParameterRole::OutputBuffer:
      if (!llvm::StringRef(expected.cType).contains(signedCType))
        return makeRVVTargetRouteError(
            llvm::Twine("elementwise arithmetic target artifact consumer "
                        "requires ABI parameter '") +
            expected.cName + "' C type '" + expected.cType +
            "' to carry signed element C type '" + signedCType + "'");
      break;
    case support::RuntimeABIParameterRole::RuntimeElementCount:
      if (expected.cType != description.vlCType)
        return makeRVVTargetRouteError(
            "elementwise arithmetic target artifact consumer requires runtime "
            "element-count ABI C type to mirror provider VL C type");
      break;
    case support::RuntimeABIParameterRole::LHSInputStride:
    case support::RuntimeABIParameterRole::RHSInputStride:
    case support::RuntimeABIParameterRole::OutputStride:
      if (expected.cType != "size_t" && expected.cType != "ptrdiff_t")
        return makeRVVTargetRouteError(
            "strided elementwise arithmetic target artifact consumer requires "
            "stride ABI C types to be size_t or ptrdiff_t integer stride "
            "parameters");
      break;
    default:
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer rejects ABI roles "
          "outside lhs, rhs, out, runtime n, and optional stride roles");
    }
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires a runtime "
        "n/AVL ABI parameter before artifact export");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires exactly "
        "one provider-built pre-loop setvl statement before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN->cName ||
      preLoopSetVL.operands.front().cType != runtimeN->cType ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to use runtime n/AVL and "
        "define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires pre-loop "
        "setvl provenance from the selected typed RVV body");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN->cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      loop.bodySteps.front().operands.size() != 1 ||
      loop.bodySteps.front().operands.front().expression !=
          expectedRemainingAVL ||
      loop.bodySteps.front().operands.front().cType != description.vlCType ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  if (isStrided) {
    if (!routeLoopContainsCallee(loop, description.stridedLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.stridedStoreIntrinsic))
      return makeRVVTargetRouteError(
          "strided elementwise arithmetic target artifact consumer requires "
          "provider-built strided loads, elementwise compute, and strided "
          "store statements before artifact export");
  } else {
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires "
          "provider-built vector loads, elementwise compute, and store "
          "statements before artifact export");
  }
  if (isMasked) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic) ||
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-built compare and masked merge statements before artifact "
          "export");
  } else if (!description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "masked compare/merge leaves for non-masked elementwise routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVElementwiseArithmeticRouteFamilyDescription(description))
    return llvm::Error::success();

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty() ||
      description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-supported, route-family, and operand-binding mirror facts "
        "before artifact export");

  llvm::StringRef expectedRuntimeABIOrder =
      isRVVStridedElementwiseArithmeticRouteFamilyOperation(
          description.operation)
          ? llvm::StringRef("lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride")
          : llvm::StringRef("lhs,rhs,out,n");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder != expectedRuntimeABIOrder ||
      description.elementTypeName.empty() || description.sew == 0 ||
      description.lmul.empty() || description.tailPolicy.empty() ||
      description.maskPolicy.empty() || description.configContractID.empty() ||
      description.setVLIntrinsic.empty() ||
      description.vectorLoadIntrinsic.empty() || description.intrinsic.empty() ||
      description.storeIntrinsic.empty() || description.resultName.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived runtime AVL/VL, dtype, SEW, LMUL, policy, "
        "config, intrinsic, and result facts before artifact export");
  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.sourceVectorTypeName.empty() ||
      !description.sourceVectorCType.empty() ||
      !description.sourceVectorLoadIntrinsic.empty() ||
      !description.conversionRelation.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "scalar-broadcast or conversion/source route facts");

  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  if (isMasked) {
    if (description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty() ||
        description.maskName.empty() || description.maskRole.empty() ||
        description.maskSource.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty())
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-derived mask, compare, merge, inactive-lane, and "
          "passthrough facts before artifact export");
  } else if (!description.maskName.empty() || !description.maskRole.empty() ||
             !description.maskSource.empty() ||
             !description.inactiveLaneContract.empty() ||
             !description.maskedPassthroughLayout.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale mask "
        "metadata for non-masked elementwise routes");
  }
  if (isStrided) {
    if (description.stridedLoadIntrinsic.empty() ||
        description.stridedStoreIntrinsic.empty() ||
        description.stridedMemoryLayout.empty() ||
        description.lhsStrideSource.empty() ||
        description.rhsStrideSource.empty() ||
        description.outStrideSource.empty())
      return makeRVVTargetRouteError(
          "strided elementwise arithmetic target artifact consumer requires "
          "provider-derived strided load/store and stride-source facts before "
          "artifact export");
  } else if (!description.stridedLoadIntrinsic.empty() ||
             !description.stridedStoreIntrinsic.empty() ||
             !description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty() ||
             !description.outStrideSource.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "strided metadata for non-strided elementwise routes");
  }

  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteTypeMappings(route,
                                                            description))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteABIMappings(route, description))
    return error;
  return validateRVVElementwiseArithmeticRouteStatementPlan(route,
                                                            description);
}

llvm::Error validateRVVConversionDtypePolicyRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived VL and result vector C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVConversionDtypePolicyWideningRouteFamilyOperation(
          description.operation)) {
    if (description.sourceVectorTypeName.empty() ||
        description.sourceVectorCType.empty())
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires provider-derived source vector type mapping facts before "
          "artifact export");
    if (!routeHasTypeMapping(route, description.sourceVectorTypeName,
                             description.sourceVectorCType))
      return makeRVVTargetRouteError(
          llvm::Twine("widening conversion dtype-policy target artifact "
                      "consumer requires rebuilt provider route type mapping "
                      "'") +
          description.sourceVectorTypeName + "' -> '" +
          description.sourceVectorCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer requires "
          "pre-loop statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression !=
          description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  if (isRVVConversionDtypePolicyScalarBroadcastRouteFamilyOperation(
          description.operation)) {
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "scalar-broadcast conversion dtype-policy target artifact consumer "
          "requires provider-built vector load, RHS scalar broadcast, "
          "elementwise compute, and store statements before artifact export");
  } else if (isRVVConversionDtypePolicyWideningRouteFamilyOperation(
                 description.operation)) {
    if (!routeLoopContainsCallee(loop,
                                 description.sourceVectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires provider-built source load, widening conversion, and "
          "store statements before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVConversionDtypePolicyRouteFamilyOperation(description.operation))
    return llvm::Error::success();

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires a "
        "provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires provider "
        "route operand binding facts before artifact export");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder.empty() || description.sew == 0 ||
      description.lmul.empty() || description.tailPolicy.empty() ||
      description.maskPolicy.empty() || description.setVLIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty() ||
      description.resultName.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived runtime, dtype, policy, intrinsic, and result facts "
        "before artifact export");

  if (isRVVConversionDtypePolicyScalarBroadcastRouteFamilyOperation(
          description.operation)) {
    if (description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
        description.vectorLoadIntrinsic.empty() ||
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast conversion dtype-policy target artifact consumer "
          "requires provider-derived scalar-broadcast plan, vector load, and "
          "RHS broadcast facts before artifact export");
    if (description.sourceSEW != 0 || !description.sourceLMUL.empty() ||
        !description.sourceVectorTypeName.empty() ||
        !description.sourceVectorCType.empty() ||
        !description.sourceVectorLoadIntrinsic.empty() ||
        !description.conversionRelation.empty() ||
        !description.wideningConversionRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast conversion dtype-policy target artifact consumer "
          "must reject stale widening source/conversion facts");
  } else {
    if (description.wideningConversionRouteFamilyPlanID.empty() ||
        description.sourceSEW == 0 || description.sourceLMUL.empty() ||
        description.sourceVectorTypeName.empty() ||
        description.sourceVectorCType.empty() ||
        description.sourceVectorLoadIntrinsic.empty() ||
        description.conversionRelation.empty())
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires provider-derived source dtype, source vector type, "
          "source load, conversion relation, and widening plan facts before "
          "artifact export");
    if (description.sourceSEW >= description.sew)
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires source SEW to be smaller than result SEW");
  }

  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteABIMappings(route, description))
    return error;
  return validateRVVConversionDtypePolicyRouteStatementPlan(route, description);
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires at "
        "least one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "runtime scalar splat-store target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIParameters.size() != 3)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived ABI order rhs_scalar,out,n before artifact export");
  const support::RuntimeABIParameter &rhsScalar =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter &out = description.runtimeABIParameters[1];
  const support::RuntimeABIParameter &runtimeN =
      description.runtimeABIParameters[2];

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires exactly "
        "one provider-built pre-loop setvl statement before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN.cName ||
      preLoopSetVL.operands.front().cType != runtimeN.cType ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to use runtime n/AVL and "
        "define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "pre-loop setvl to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN.cName ||
      loop.upperBound.cType != runtimeN.cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  if (loop.bodySteps.size() != 3)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop setvl, runtime scalar splat, and store "
        "statements before artifact export");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &loopSetVL =
      loop.bodySteps[0];
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (loopSetVL.callee != description.setVLIntrinsic ||
      loopSetVL.operands.size() != 1 ||
      loopSetVL.operands.front().expression != expectedRemainingAVL ||
      loopSetVL.operands.front().cType != description.vlCType ||
      !stepHasResult(loopSetVL, description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &splat = loop.bodySteps[1];
  if (splat.callee != description.rhsBroadcastIntrinsic ||
      splat.operands.size() != 2 ||
      splat.operands[0].expression != rhsScalar.cName ||
      splat.operands[0].cType != rhsScalar.cType ||
      splat.operands[1].expression != description.emitCLoopVLName ||
      splat.operands[1].cType != description.vlCType ||
      !stepHasResult(splat, description.resultName, description.vectorCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built runtime scalar splat statement to consume rhs_scalar "
        "and per-iteration VL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &store = loop.bodySteps[2];
  const std::string expectedOutPointer =
      (llvm::StringRef(out.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (store.callee != description.storeIntrinsic ||
      store.operands.size() != 3 ||
      store.operands[0].expression != expectedOutPointer ||
      store.operands[0].cType != out.cType ||
      store.operands[1].expression != description.resultName ||
      store.operands[1].cType != description.vectorCType ||
      store.operands[2].expression != description.emitCLoopVLName ||
      store.operands[2].cType != description.vlCType)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built store statement to consume out+offset, splat vector, "
        "and per-iteration VL");

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "runtime scalar splat-store target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVRuntimeScalarSplatStoreRouteFamilyOperation(description.operation))
    return llvm::Error::success();

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty() ||
      description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-supported, route-family, and operand-binding mirror facts "
        "before artifact export");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder != "rhs_scalar,out,n" ||
      description.sew == 0 || description.lmul.empty() ||
      description.tailPolicy.empty() || description.maskPolicy.empty() ||
      description.setVLIntrinsic.empty() ||
      description.rhsBroadcastIntrinsic.empty() ||
      description.storeIntrinsic.empty() || description.resultName.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived runtime AVL/VL, dtype, policy, splat, store, and "
        "result facts before artifact export");
  if (!description.intrinsic.empty() ||
      !description.sourceVectorTypeName.empty() ||
      !description.sourceVectorCType.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer must reject stale "
        "elementwise or conversion route facts");

  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteTypeMappings(route,
                                                              description))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteABIMappings(route,
                                                             description))
    return error;
  return validateRVVRuntimeScalarSplatStoreRouteStatementPlan(route,
                                                              description);
}

llvm::Error validateRVVSegment2RouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires provider-derived "
        "required_header_declarations before accepting the route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires at least one provider "
        "route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "segment2 target artifact consumer saw an empty provider route "
          "header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("segment2 target artifact consumer requires rebuilt "
                      "provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2RouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires provider-derived VL, "
        "vector, and C type mapping facts before artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("segment2 target artifact consumer requires rebuilt "
                    "provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("segment2 target artifact consumer requires rebuilt "
                    "provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVComputedMaskSegment2RouteFamilyOperation(description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2 target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask segment2 target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2RouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("segment2 target artifact consumer requires rebuilt "
                    "provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("segment2 target artifact consumer requires rebuilt "
                      "provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("segment2 target artifact consumer requires rebuilt "
                      "provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2RouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires provider-built pre-loop "
        "setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires rebuilt provider route "
        "pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2 target artifact consumer requires pre-loop statements to "
          "carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires exactly one provider-built "
        "runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires provider-built loop "
        "bounds and step to mirror runtime AVL/VL route facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression != description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires provider-built loop upper "
        "bound to use the runtime n/AVL ABI parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires provider-built loop setvl "
        "statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2 target artifact consumer requires loop statements to "
          "carry selected typed RVV source provenance");

  if (isRVVComputedMaskSegment2RouteFamilyOperation(description.operation) &&
      !routeLoopContainsCallee(loop, description.compareIntrinsic))
    return makeRVVTargetRouteError(
        "computed-mask segment2 target artifact consumer requires "
        "provider-built compare/mask statement facts before artifact export");

  switch (description.operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentFieldExtractIntrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 load target artifact consumer requires "
          "provider-built segment load, tuple, extract, and store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 store target artifact consumer requires "
          "provider-built tuple and masked segment-store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    if (description.intrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2 update target artifact consumer requires "
          "provider-derived update arithmetic facts before artifact export");
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 update target artifact consumer requires "
          "provider-built arithmetic, tuple, and masked segment-store "
          "statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2DeinterleaveUnitStore:
    if (!routeLoopContainsCallee(loop, description.segmentLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentFieldExtractIntrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "segment2 deinterleave target artifact consumer requires "
          "provider-built segment load, extract, and field-store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2InterleaveUnitLoad:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "segment2 interleave target artifact consumer requires "
          "provider-built tuple and segment-store statements");
    break;
  default:
    llvm_unreachable("validated non-segment2 operation as segment2");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2RoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVSegment2RouteFamilyOperation(description.operation))
    return llvm::Error::success();

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("segment2 target artifact consumer requires rebuilt "
                    "provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (isRVVComputedMaskSegment2RouteFamilyOperation(description.operation)) {
    if (description.computedMaskMemoryRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2 target artifact consumer requires a "
          "provider-derived computed-mask route-family plan mirror before "
          "artifact export");
  } else if (isRVVPlainSegment2RouteFamilyOperation(description.operation)) {
    if (description.segment2MemoryRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "segment2 target artifact consumer requires a provider-derived "
          "segment2 route-family plan mirror before artifact export");
  }
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires a provider-supported "
        "mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "segment2 target artifact consumer requires provider route operand "
        "binding facts before artifact export");

  if (llvm::Error error = validateRVVSegment2RouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVSegment2RouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVSegment2RouteABIMappings(route, description))
    return error;
  return validateRVVSegment2RouteStatementPlan(route, description);
}

llvm::Error validateRVVMAccRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived "
        "required_header_declarations before accepting the route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires at least one provider route "
        "header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer saw an empty provider route header "
          "declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt "
                      "provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived VL, vector, "
        "and C type mapping facts before artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask MAcc target artifact consumer requires "
                      "rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                      "route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                      "route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built pre-loop setvl "
        "statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires rebuilt provider route "
        "pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer requires pre-loop statements to "
          "carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires exactly one provider-built "
        "runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop bounds "
        "and step to mirror runtime AVL/VL route facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression != description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop upper "
        "bound to use the runtime n/AVL ABI parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop setvl "
        "statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer requires loop statements to carry "
          "selected typed RVV source provenance");

  if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built vector load, "
        "MAcc arithmetic, and store statements before artifact export");

  if (isRVVScalarBroadcastMAccRouteFamilyOperation(description.operation) &&
      !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
    return makeRVVTargetRouteError(
        "scalar-broadcast MAcc target artifact consumer requires "
        "provider-built RHS scalar broadcast statement facts before artifact "
        "export");

  if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-built compare statement facts before artifact export");
    if (!description.maskedMergeIntrinsic.empty() &&
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-built masked-merge statement facts before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVMAccRouteFamilyOperation(description.operation))
    return llvm::Error::success();

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires a provider-supported mirror "
        "label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider route operand "
        "binding facts before artifact export");
  if (description.maccAccumulatorLayout.empty() ||
      description.maccResultLayout.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived accumulator "
        "and result layout facts before artifact export");

  if (description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
    if (description.plainMAccRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "plain MAcc target artifact consumer requires a provider-derived "
          "plain MAcc route-family plan mirror before artifact export");
  } else if (isRVVScalarBroadcastMAccRouteFamilyOperation(
                 description.operation)) {
    if (description.scalarBroadcastMAccRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast MAcc target artifact consumer requires a "
          "provider-derived scalar-broadcast MAcc route-family plan mirror "
          "before artifact export");
    if (description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast MAcc target artifact consumer requires a "
          "provider-derived RHS scalar broadcast intrinsic before artifact "
          "export");
  } else if (isRVVComputedMaskMAccRouteFamilyOperation(
                 description.operation)) {
    if (description.accumulationRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires a "
          "provider-derived accumulation route-family plan mirror before "
          "artifact export");
    if (description.accumulationComputeSuffix.empty() ||
        description.accumulationMaskProducerSource.empty() ||
        description.accumulationAccumulatorContract.empty() ||
        description.accumulationResultContract.empty() ||
        description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-derived mask, passthrough, compare, accumulator, "
          "and result facts before artifact export");
  }

  if (llvm::Error error = validateRVVMAccRouteHeaders(route, description))
    return error;
  if (llvm::Error error = validateRVVMAccRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error = validateRVVMAccRouteABIMappings(route, description))
    return error;
  return validateRVVMAccRouteStatementPlan(route, description);
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "at least one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer saw an "
          "empty provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  if (description.standaloneReductionSourceVectorTypeName.empty() ||
      description.standaloneReductionSourceVectorCType.empty() ||
      description.standaloneReductionScalarResultVectorTypeName.empty() ||
      description.standaloneReductionScalarResultVectorCType.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived source and scalar-result vector type mapping facts "
        "before artifact export");
  if (!routeHasTypeMapping(
          route, description.standaloneReductionSourceVectorTypeName,
          description.standaloneReductionSourceVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route source type "
                    "mapping '") +
        description.standaloneReductionSourceVectorTypeName + "' -> '" +
        description.standaloneReductionSourceVectorCType + "'");
  if (!routeHasTypeMapping(
          route, description.standaloneReductionScalarResultVectorTypeName,
          description.standaloneReductionScalarResultVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route scalar-result "
                    "type mapping '") +
        description.standaloneReductionScalarResultVectorTypeName + "' -> '" +
        description.standaloneReductionScalarResultVectorCType + "'");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-derived mask type mapping facts before "
          "artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask standalone reduction/accumulation target "
                      "artifact consumer requires rebuilt provider route type "
                      "mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route ABI mapping "
                    "count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route ABI "
                      "mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route ABI "
                      "mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "rebuilt provider route pre-loop setvl statement to define the "
        "full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer "
          "requires pre-loop statements to carry selected typed RVV source "
          "provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "exactly one provider-built runtime AVL/VL loop before artifact "
        "export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression !=
          description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer "
          "requires loop statements to carry selected typed RVV source "
          "provenance");

  if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic) ||
      !routeLoopContainsCallee(loop, description.scalarSeedSplatIntrinsic))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built vector load, reduction, seed splat, and store "
        "statements before artifact export");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-built compare statement facts before "
          "artifact export");
    if (!routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-built inactive-lane merge statement "
          "facts before artifact export");
    if (isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation) &&
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask standalone reduction/accumulation "
          "target artifact consumer requires provider-built RHS scalar splat "
          "statement facts before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVStandaloneReductionAccumulationRouteFamilyOperation(
          description.operation))
    return llvm::Error::success();

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "a provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider route operand binding facts before artifact export");
  if (description.standaloneReductionRouteFamilyPlanID.empty() ||
      description.standaloneReductionScalarResultRuntimeBoundary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived standalone reduction family and scalar-result "
        "boundary mirrors before artifact export");
  if (description.reductionAccumulatorLayout.empty() ||
      description.reductionResultLayout.empty() ||
      description.reductionStoreVL.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived reduction accumulator, result, and store-VL facts "
        "before artifact export");
  if (description.scalarSeedSplatIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived scalar seed, reduction intrinsic, and store facts "
        "before artifact export");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneZeroingRequirement.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-derived mask, compare, and "
          "inactive-lane facts before artifact export");
  }
  if (isRVVComputedMaskStandaloneReductionAccumulationRouteFamilyOperation(
          description.operation)) {
    if (description.accumulationRouteFamilyPlanID.empty() ||
        description.accumulationComputeSuffix.empty() ||
        description.accumulationMaskProducerSource.empty() ||
        description.accumulationAccumulatorContract.empty() ||
        description.accumulationResultContract.empty() ||
        description.accumulationScalarCarryContract.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone accumulation target artifact consumer "
          "requires provider-derived accumulation and scalar-carry facts "
          "before artifact export");
    if (isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation) &&
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask standalone reduction/accumulation "
          "target artifact consumer requires a provider-derived RHS scalar "
          "splat intrinsic before artifact export");
  }

  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteHeaders(route,
                                                                description))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteTypeMappings(
              route, description))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteABIMappings(
              route, description))
    return error;
  return validateRVVStandaloneReductionAccumulationRouteStatementPlan(
      route, description);
}

struct RVVSelectedVariantRouteValidation {
  conversion::emitc::TCRVEmitCLowerableRoute route;
  plugin::rvv::RVVSelectedBodyEmitCRouteDescription description;
};

llvm::Error validateRVVRouteMetadataMirrorsSelectedBody(
    const TargetArtifactCandidate &candidate,
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::StringRef routeID = lookupCandidateMetadataValue(
      candidate, plugin::rvv::getRVVEmitCLowerableRouteMetadataName());
  llvm::StringRef selectedBodyOperation = lookupCandidateMetadataValue(
      candidate, plugin::rvv::getRVVSelectedBodyOperationMetadataName());
  if (routeID.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry ") +
        plugin::rvv::getRVVEmitCLowerableRouteMetadataName() +
        " provenance");
  if (route.getRouteID() != routeID)
    return makeRVVTargetRouteError(
        llvm::Twine("candidate ") +
        plugin::rvv::getRVVEmitCLowerableRouteMetadataName() +
        " provenance must mirror selected typed RVV body route '" +
        route.getRouteID() + "' but was '" + routeID + "'");

  if (selectedBodyOperation.empty())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry ") +
        plugin::rvv::getRVVSelectedBodyOperationMetadataName() +
        " provenance");
  llvm::StringRef expectedOperation =
      plugin::rvv::stringifyRVVSelectedBodyOperationKind(
          description.operation);
  if (selectedBodyOperation != expectedOperation)
    return makeRVVTargetRouteError(
        llvm::Twine("candidate ") +
        plugin::rvv::getRVVSelectedBodyOperationMetadataName() +
        " provenance must mirror selected typed RVV body operation '" +
        expectedOperation + "' but was '" + selectedBodyOperation + "'");
  llvm::StringRef comparePredicateKind = lookupCandidateMetadataValue(
      candidate, "tcrv_rvv.compare_predicate_kind");
  if (!description.comparePredicateKind.empty()) {
    if (comparePredicateKind.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "tcrv_rvv.compare_predicate_kind provenance");
    if (comparePredicateKind != description.comparePredicateKind)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate tcrv_rvv.compare_predicate_kind provenance "
                      "must mirror selected typed RVV body predicate '") +
          description.comparePredicateKind + "' but was '" +
          comparePredicateKind + "'");
  } else if (!comparePredicateKind.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry compare predicate mirrors for a "
        "selected typed RVV body route without a compare predicate");
  }
  llvm::StringRef routeOperandBindingPlan = lookupCandidateMetadataValue(
      candidate, "tcrv_rvv.route_operand_binding_plan");
  llvm::StringRef routeOperandBindingOperands = lookupCandidateMetadataValue(
      candidate, "tcrv_rvv.route_operand_binding_operands");
  if (!description.routeOperandBindingPlanID.empty()) {
    if (routeOperandBindingPlan.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "tcrv_rvv.route_operand_binding_plan provenance");
    if (routeOperandBindingPlan != description.routeOperandBindingPlanID)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate tcrv_rvv.route_operand_binding_plan "
                      "provenance must mirror selected typed RVV body binding "
                      "plan '") +
          description.routeOperandBindingPlanID + "' but was '" +
          routeOperandBindingPlan + "'");
    if (routeOperandBindingOperands.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "tcrv_rvv.route_operand_binding_operands provenance");
    if (routeOperandBindingOperands != description.routeOperandBindingSummary)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate tcrv_rvv.route_operand_binding_operands "
                      "provenance must mirror selected typed RVV body binding "
                      "summary '") +
          description.routeOperandBindingSummary + "' but was '" +
          routeOperandBindingOperands + "'");
  } else if (!routeOperandBindingPlan.empty() ||
             !routeOperandBindingOperands.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry route operand binding mirrors for a "
        "selected typed RVV body route without a binding plan");
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV body provider support"))
    return error;

  if (isRVVElementwiseArithmeticRouteFamilyDescription(description)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.elementwise_arithmetic_route_family_plan",
            description.elementwiseArithmeticRouteFamilyPlanID,
            "selected typed RVV elementwise arithmetic route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.memory_form",
            plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
                description.memoryForm),
            "selected typed RVV elementwise memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.target_leaf_profile",
            description.targetLeafProfile,
            "selected typed RVV elementwise target leaf profile"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_control_plan",
            description.runtimeControlPlanID,
            "selected typed RVV elementwise runtime AVL/VL control plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_abi_order",
            description.runtimeABIOrder,
            "selected typed RVV elementwise runtime ABI order"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.required_header_declarations",
            description.requiredHeaderDeclarations,
            "selected typed RVV elementwise route header requirements"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.c_type_mapping",
            description.cTypeMappingSummary,
            "selected typed RVV elementwise route type mapping summary"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form",
            description.sourceMemoryForm,
            "selected typed RVV elementwise source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form",
            description.destinationMemoryForm,
            "selected typed RVV elementwise destination memory form"))
      return error;

    if (isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
            description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_role", description.maskRole,
              "selected typed RVV masked elementwise mask role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_source", description.maskSource,
              "selected typed RVV masked elementwise mask source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.inactive_lane_contract",
              description.inactiveLaneContract,
              "selected typed RVV masked elementwise inactive lane contract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.masked_passthrough_layout",
              description.maskedPassthroughLayout,
              "selected typed RVV masked elementwise passthrough layout"))
        return error;
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_role", "",
              "selected typed RVV masked elementwise mask role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_source", "",
              "selected typed RVV masked elementwise mask source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.inactive_lane_contract", "",
              "selected typed RVV masked elementwise inactive lane contract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.masked_passthrough_layout", "",
              "selected typed RVV masked elementwise passthrough layout"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", "",
            "selected typed RVV masked elementwise mask memory form"))
      return error;

    if (isRVVStridedElementwiseArithmeticRouteFamilyOperation(
            description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.strided_memory_layout",
              description.stridedMemoryLayout,
              "selected typed RVV strided elementwise memory layout"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.lhs_stride_source",
              description.lhsStrideSource,
              "selected typed RVV strided elementwise lhs stride source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.rhs_stride_source",
              description.rhsStrideSource,
              "selected typed RVV strided elementwise rhs stride source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.out_stride_source",
              description.outStrideSource,
              "selected typed RVV strided elementwise output stride source"))
        return error;
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.strided_memory_layout", "",
              "selected typed RVV strided elementwise memory layout"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.lhs_stride_source", "",
              "selected typed RVV strided elementwise lhs stride source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.rhs_stride_source", "",
              "selected typed RVV strided elementwise rhs stride source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.out_stride_source", "",
              "selected typed RVV strided elementwise output stride source"))
        return error;
    }
  } else if (llvm::Error error = requireCandidateMetadataMirror(
                 candidate,
                 "tcrv_rvv.elementwise_arithmetic_route_family_plan", "",
                 "selected typed RVV elementwise arithmetic route-family "
                 "plan")) {
    return error;
  }

  if (isRVVCompareSelectMaskRouteFamilyOperation(description.operation)) {
    if (isRVVPlainCompareSelectMaskRouteFamilyOperation(
            description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.plain_compare_select_route_family_plan",
              description.plainCompareSelectRouteFamilyPlanID,
              "selected typed RVV plain compare-select route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_select_route_family_plan", "",
              "selected typed RVV computed-mask select route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
              "selected typed RVV computed-mask memory route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_tail_policy_route_family_plan", "",
              "selected typed RVV mask/tail policy route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_tail_policy_owner", "",
              "selected typed RVV mask/tail policy owner"))
        return error;
    } else if (isRVVComputedMaskSelectRouteFamilyOperation(
                   description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_select_route_family_plan",
              description.computedMaskSelectRouteFamilyPlanID,
              "selected typed RVV computed-mask select route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate,
              "tcrv_rvv.computed_mask_select_mask_producer_source",
              description.computedMaskSelectMaskProducerSource,
              "selected typed RVV computed-mask select producer source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.plain_compare_select_route_family_plan", "",
              "selected typed RVV plain compare-select route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
              "selected typed RVV computed-mask memory route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
              description.maskTailPolicyRouteFamilyPlanID,
              "selected typed RVV mask/tail policy route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_tail_policy_owner",
              description.maskTailPolicyOwner,
              "selected typed RVV mask/tail policy owner"))
        return error;
    } else if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
                   description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_memory_route_family_plan",
              description.computedMaskMemoryRouteFamilyPlanID,
              "selected typed RVV computed-mask memory route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate,
              "tcrv_rvv.computed_mask_memory_mask_producer_source",
              description.computedMaskMemoryMaskProducerSource,
              "selected typed RVV computed-mask memory producer source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.plain_compare_select_route_family_plan", "",
              "selected typed RVV plain compare-select route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_select_route_family_plan", "",
              "selected typed RVV computed-mask select route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
              description.maskTailPolicyRouteFamilyPlanID,
              "selected typed RVV mask/tail policy route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_tail_policy_owner",
              description.maskTailPolicyOwner,
              "selected typed RVV mask/tail policy owner"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.memory_form",
            plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
                description.memoryForm),
            "selected typed RVV compare/select mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_control_plan",
            description.runtimeControlPlanID,
            "selected typed RVV compare/select mask runtime AVL/VL control "
            "plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_abi_order",
            description.runtimeABIOrder,
            "selected typed RVV compare/select mask runtime ABI order"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.required_header_declarations",
            description.requiredHeaderDeclarations,
            "selected typed RVV compare/select mask route header requirements"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.c_type_mapping",
            description.cTypeMappingSummary,
            "selected typed RVV compare/select mask route type mapping "
            "summary"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV compare/select mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV compare/select mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form",
            description.maskMemoryForm,
            "selected typed RVV compare/select mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form",
            description.sourceMemoryForm,
            "selected typed RVV compare/select mask source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form",
            description.destinationMemoryForm,
            "selected typed RVV compare/select mask destination memory form"))
      return error;
    if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
            description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.select_layout",
              description.selectLayout,
              "selected typed RVV compare/select selected-value layout"))
        return error;
      if (description.operation ==
          plugin::rvv::RVVSelectedBodyOperationKind::
              RuntimeScalarDualCompareMaskAndSelect) {
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.mask_composition",
                description.maskComposition,
                "selected typed RVV dual compare mask composition"))
          return error;
      }
    }
    if (isRVVPlainCompareSelectMaskRouteFamilyOperation(
            description.operation) ||
        isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
            description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.inactive_lane_contract",
              description.inactiveLaneContract,
              "selected typed RVV compare/select mask inactive lane "
              "contract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.masked_passthrough_layout",
              description.maskedPassthroughLayout,
              "selected typed RVV compare/select mask passthrough layout"))
        return error;
    }
    if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
            description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.masked_memory_layout",
              description.indexedMemoryLayout,
              "selected typed RVV computed-mask memory layout"))
        return error;
    }
  }

  if (isRVVConversionDtypePolicyRouteFamilyOperation(description.operation)) {
    if (isRVVConversionDtypePolicyScalarBroadcastRouteFamilyOperation(
            description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate,
              "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
              description.scalarBroadcastElementwiseRouteFamilyPlanID,
              "selected typed RVV scalar-broadcast elementwise route-family "
              "plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.widening_conversion_route_family_plan", "",
              "selected typed RVV widening conversion route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.source_sew", "",
              "selected typed RVV widening conversion source SEW"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.source_lmul", "",
              "selected typed RVV widening conversion source LMUL"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dest_sew", "",
              "selected typed RVV widening conversion destination SEW"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dest_lmul", "",
              "selected typed RVV widening conversion destination LMUL"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.conversion_relation", "",
              "selected typed RVV widening conversion relation"))
        return error;
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.widening_conversion_route_family_plan",
              description.wideningConversionRouteFamilyPlanID,
              "selected typed RVV widening conversion route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate,
              "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan", "",
              "selected typed RVV scalar-broadcast elementwise route-family "
              "plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.source_sew",
              llvm::Twine(description.sourceSEW).str(),
              "selected typed RVV widening conversion source SEW"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.source_lmul", description.sourceLMUL,
              "selected typed RVV widening conversion source LMUL"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dest_sew",
              llvm::Twine(description.sew).str(),
              "selected typed RVV widening conversion destination SEW"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.dest_lmul", description.lmul,
              "selected typed RVV widening conversion destination LMUL"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.conversion_relation",
              description.conversionRelation,
              "selected typed RVV widening conversion relation"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.memory_form",
            plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
                description.memoryForm),
            "selected typed RVV conversion dtype-policy memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_control_plan",
            description.runtimeControlPlanID,
            "selected typed RVV conversion dtype-policy runtime AVL/VL "
            "control plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_abi_order",
            description.runtimeABIOrder,
            "selected typed RVV conversion dtype-policy runtime ABI order"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.required_header_declarations",
            description.requiredHeaderDeclarations,
            "selected typed RVV conversion dtype-policy route header "
            "requirements"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.c_type_mapping",
            description.cTypeMappingSummary,
            "selected typed RVV conversion dtype-policy route type mapping "
            "summary"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate,
            "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan", "",
            "selected typed RVV scalar-broadcast elementwise route-family "
            "plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.widening_conversion_route_family_plan", "",
            "selected typed RVV widening conversion route-family plan"))
      return error;
  }

  if (isRVVSegment2RouteFamilyOperation(description.operation)) {
    if (isRVVPlainSegment2RouteFamilyOperation(description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment2_memory_route_family_plan",
              description.segment2MemoryRouteFamilyPlanID,
              "selected typed RVV segment2 route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
              "selected typed RVV computed-mask memory route-family plan"))
        return error;
    } else if (isRVVComputedMaskSegment2RouteFamilyOperation(
                   description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.segment2_memory_route_family_plan", "",
              "selected typed RVV segment2 route-family plan"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.memory_form",
            plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
                description.memoryForm),
            "selected typed RVV segment2 memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_control_plan",
            description.runtimeControlPlanID,
            "selected typed RVV runtime AVL/VL control plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_abi_order",
            description.runtimeABIOrder,
            "selected typed RVV runtime ABI order"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.required_header_declarations",
            description.requiredHeaderDeclarations,
            "selected typed RVV route header requirements"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.c_type_mapping",
            description.cTypeMappingSummary,
            "selected typed RVV route type mapping summary"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_memory_layout",
            description.segmentMemoryLayout,
            "selected typed RVV segment2 memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form",
            description.sourceMemoryForm,
            "selected typed RVV segment2 source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form",
            description.destinationMemoryForm,
            "selected typed RVV segment2 destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment_count",
            llvm::Twine(description.segmentCount).str(),
            "selected typed RVV segment2 count"))
      return error;
    if (isRVVComputedMaskSegment2RouteFamilyOperation(description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_memory_route_family_plan",
              description.computedMaskMemoryRouteFamilyPlanID,
              "selected typed RVV computed-mask memory route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
              description.computedMaskMemoryMaskProducerSource,
              "selected typed RVV computed-mask segment2 producer source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_role", description.maskRole,
              "selected typed RVV computed-mask segment2 mask role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_source", description.maskSource,
              "selected typed RVV computed-mask segment2 mask source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_memory_form",
              description.maskMemoryForm,
              "selected typed RVV computed-mask segment2 mask memory form"))
        return error;
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
              "", "selected typed RVV computed-mask segment2 producer source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_role", "",
              "selected typed RVV computed-mask segment2 mask role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_source", "",
              "selected typed RVV computed-mask segment2 mask source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_memory_form", "",
              "selected typed RVV computed-mask segment2 mask memory form"))
        return error;
    }
  } else if (llvm::Error error = requireCandidateMetadataMirror(
                 candidate, "tcrv_rvv.segment2_memory_route_family_plan", "",
                 "selected typed RVV segment2 route-family plan")) {
    return error;
  }

  if (isRVVMAccRouteFamilyOperation(description.operation)) {
    if (description.operation ==
        plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.plain_macc_route_family_plan",
              description.plainMAccRouteFamilyPlanID,
              "selected typed RVV plain MAcc route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
              "", "selected typed RVV scalar-broadcast MAcc route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_route_family_plan", "",
              "selected typed RVV accumulation route-family plan"))
        return error;
    } else if (isRVVScalarBroadcastMAccRouteFamilyOperation(
                   description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
              description.scalarBroadcastMAccRouteFamilyPlanID,
              "selected typed RVV scalar-broadcast MAcc route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.plain_macc_route_family_plan", "",
              "selected typed RVV plain MAcc route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_route_family_plan", "",
              "selected typed RVV accumulation route-family plan"))
        return error;
    } else if (isRVVComputedMaskMAccRouteFamilyOperation(
                   description.operation)) {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_route_family_plan",
              description.accumulationRouteFamilyPlanID,
              "selected typed RVV computed-mask MAcc route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_compute_suffix",
              description.accumulationComputeSuffix,
              "selected typed RVV computed-mask MAcc compute suffix"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_mask_producer_source",
              description.accumulationMaskProducerSource,
              "selected typed RVV computed-mask MAcc mask producer"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_accumulator_contract",
              description.accumulationAccumulatorContract,
              "selected typed RVV computed-mask MAcc accumulator contract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_result_contract",
              description.accumulationResultContract,
              "selected typed RVV computed-mask MAcc result contract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_scalar_carry_contract",
              description.accumulationScalarCarryContract,
              "selected typed RVV computed-mask MAcc scalar carry contract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.plain_macc_route_family_plan", "",
              "selected typed RVV plain MAcc route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
              "", "selected typed RVV scalar-broadcast MAcc route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_role", description.maskRole,
              "selected typed RVV computed-mask MAcc mask role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_source", description.maskSource,
              "selected typed RVV computed-mask MAcc mask source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_memory_form",
              description.maskMemoryForm,
              "selected typed RVV computed-mask MAcc mask memory form"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.inactive_lane_contract",
              description.inactiveLaneContract,
              "selected typed RVV computed-mask MAcc inactive lane contract"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.masked_passthrough_layout",
              description.maskedPassthroughLayout,
              "selected typed RVV computed-mask MAcc passthrough layout"))
        return error;
    }
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.memory_form",
            plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
                description.memoryForm),
            "selected typed RVV MAcc memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_control_plan",
            description.runtimeControlPlanID,
            "selected typed RVV MAcc runtime AVL/VL control plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_abi_order",
            description.runtimeABIOrder,
            "selected typed RVV MAcc runtime ABI order"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.required_header_declarations",
            description.requiredHeaderDeclarations,
            "selected typed RVV MAcc route header requirements"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.c_type_mapping",
            description.cTypeMappingSummary,
            "selected typed RVV MAcc route type mapping summary"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.macc_accumulator_layout",
            description.maccAccumulatorLayout,
            "selected typed RVV MAcc accumulator layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.macc_result_layout",
            description.maccResultLayout,
            "selected typed RVV MAcc result layout"))
      return error;
  }

  if (isRVVStandaloneReductionAccumulationRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.standalone_reduction_route_family_plan",
            description.standaloneReductionRouteFamilyPlanID,
            "selected typed RVV standalone reduction route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate,
            "tcrv_rvv.standalone_reduction_scalar_result_runtime_boundary",
            description.standaloneReductionScalarResultRuntimeBoundary,
            "selected typed RVV standalone reduction scalar result boundary"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.memory_form",
            plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
                description.memoryForm),
            "selected typed RVV standalone reduction memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_control_plan",
            description.runtimeControlPlanID,
            "selected typed RVV standalone reduction runtime AVL/VL control "
            "plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.runtime_abi_order",
            description.runtimeABIOrder,
            "selected typed RVV standalone reduction runtime ABI order"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.required_header_declarations",
            description.requiredHeaderDeclarations,
            "selected typed RVV standalone reduction route header "
            "requirements"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.c_type_mapping",
            description.cTypeMappingSummary,
            "selected typed RVV standalone reduction route type mapping "
            "summary"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.reduction_accumulator_layout",
            description.reductionAccumulatorLayout,
            "selected typed RVV standalone reduction accumulator layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.reduction_result_layout",
            description.reductionResultLayout,
            "selected typed RVV standalone reduction result layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.reduction_store_vl",
            description.reductionStoreVL,
            "selected typed RVV standalone reduction store VL"))
      return error;
    if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation)) {
      if (isRVVComputedMaskStandaloneReductionAccumulationRouteFamilyOperation(
              description.operation)) {
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_route_family_plan",
                description.accumulationRouteFamilyPlanID,
                "selected typed RVV computed-mask standalone accumulation "
                "route-family plan"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_compute_suffix",
                description.accumulationComputeSuffix,
                "selected typed RVV computed-mask standalone accumulation "
                "compute suffix"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_mask_producer_source",
                description.accumulationMaskProducerSource,
                "selected typed RVV computed-mask standalone accumulation "
                "mask producer"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_accumulator_contract",
                description.accumulationAccumulatorContract,
                "selected typed RVV computed-mask standalone accumulation "
                "accumulator contract"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_result_contract",
                description.accumulationResultContract,
                "selected typed RVV computed-mask standalone accumulation "
                "result contract"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_scalar_carry_contract",
                description.accumulationScalarCarryContract,
                "selected typed RVV computed-mask standalone accumulation "
                "scalar carry contract"))
          return error;
      } else {
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_route_family_plan", "",
                "selected typed RVV computed-mask standalone accumulation "
                "route-family plan"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_compute_suffix", "",
                "selected typed RVV computed-mask standalone accumulation "
                "compute suffix"))
          return error;
        if (llvm::Error error = requireCandidateMetadataMirror(
                candidate, "tcrv_rvv.accumulation_mask_producer_source", "",
                "selected typed RVV computed-mask standalone accumulation "
                "mask producer"))
          return error;
      }
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_role", description.maskRole,
              "selected typed RVV computed-mask standalone reduction mask "
              "role"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_source", description.maskSource,
              "selected typed RVV computed-mask standalone reduction mask "
              "source"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.mask_memory_form",
              description.maskMemoryForm,
              "selected typed RVV computed-mask standalone reduction mask "
              "memory form"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.inactive_lane_zeroing_requirement",
              description.inactiveLaneZeroingRequirement,
              "selected typed RVV computed-mask standalone reduction "
              "inactive lane neutral requirement"))
        return error;
    } else {
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_route_family_plan", "",
              "selected typed RVV computed-mask standalone accumulation "
              "route-family plan"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_compute_suffix", "",
              "selected typed RVV computed-mask standalone accumulation "
              "compute suffix"))
        return error;
      if (llvm::Error error = requireCandidateMetadataMirror(
              candidate, "tcrv_rvv.accumulation_mask_producer_source", "",
              "selected typed RVV computed-mask standalone accumulation mask "
              "producer"))
        return error;
    }
  } else if (llvm::Error error = requireCandidateMetadataMirror(
                 candidate, "tcrv_rvv.standalone_reduction_route_family_plan",
                 "",
                 "selected typed RVV standalone reduction route-family plan")) {
    return error;
  }

  llvm::StringRef targetCapabilityProviderMirror =
      lookupCandidateMetadataValue(
          candidate, "tcrv_rvv.target_capability_provider_mirror");
  llvm::StringRef targetCapabilityLegalityMirror =
      lookupCandidateMetadataValue(
          candidate, "tcrv_rvv.target_capability_legality_mirror");
  if (!description.targetCapabilityProviderMirror.empty()) {
    if (targetCapabilityProviderMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "tcrv_rvv.target_capability_provider_mirror provenance");
    if (targetCapabilityProviderMirror !=
        llvm::StringRef(description.targetCapabilityProviderMirror))
      return makeRVVTargetRouteError(
          llvm::Twine(
              "candidate tcrv_rvv.target_capability_provider_mirror "
              "provenance must mirror selected RVV target capability facts '") +
          description.targetCapabilityProviderMirror + "' but was '" +
          targetCapabilityProviderMirror + "'");
  } else if (!targetCapabilityProviderMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry target capability provider mirrors "
        "for a selected typed RVV body route without provider facts");
  }
  if (!description.targetCapabilityLegalityMirror.empty()) {
    if (targetCapabilityLegalityMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "tcrv_rvv.target_capability_legality_mirror provenance");
    if (targetCapabilityLegalityMirror !=
        llvm::StringRef(description.targetCapabilityLegalityMirror))
      return makeRVVTargetRouteError(
          llvm::Twine(
              "candidate tcrv_rvv.target_capability_legality_mirror "
              "provenance must mirror selected RVV target capability facts '") +
          description.targetCapabilityLegalityMirror + "' but was '" +
          targetCapabilityLegalityMirror + "'");
  } else if (!targetCapabilityLegalityMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry target capability legality mirrors "
        "for a selected typed RVV body route without provider facts");
  }
  llvm::StringRef selectedDispatchCaseMirror = lookupCandidateMetadataValue(
      candidate, "tcrv_rvv.selected_dispatch_case_mirror");
  llvm::StringRef selectedDispatchFallbackMirror = lookupCandidateMetadataValue(
      candidate, "tcrv_rvv.selected_dispatch_fallback_mirror");
  if (!description.selectedDispatchCaseMirror.empty()) {
    if (selectedDispatchCaseMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "tcrv_rvv.selected_dispatch_case_mirror provenance");
    if (selectedDispatchCaseMirror !=
        llvm::StringRef(description.selectedDispatchCaseMirror))
      return makeRVVTargetRouteError(
          llvm::Twine("candidate tcrv_rvv.selected_dispatch_case_mirror "
                      "provenance must mirror selected dispatch case facts '") +
          description.selectedDispatchCaseMirror + "' but was '" +
          selectedDispatchCaseMirror + "'");
  } else if (!selectedDispatchCaseMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry selected dispatch case mirrors for "
        "a selected typed RVV body route without dispatch-case envelope facts");
  }
  if (!description.selectedDispatchFallbackMirror.empty()) {
    if (selectedDispatchFallbackMirror.empty())
      return makeRVVTargetRouteError(
          "candidate metadata must carry "
          "tcrv_rvv.selected_dispatch_fallback_mirror provenance");
    if (selectedDispatchFallbackMirror !=
        llvm::StringRef(description.selectedDispatchFallbackMirror))
      return makeRVVTargetRouteError(
          llvm::Twine(
              "candidate tcrv_rvv.selected_dispatch_fallback_mirror "
              "provenance must mirror selected dispatch fallback facts '") +
          description.selectedDispatchFallbackMirror + "' but was '" +
          selectedDispatchFallbackMirror + "'");
  } else if (!selectedDispatchFallbackMirror.empty()) {
    return makeRVVTargetRouteError(
        "candidate metadata must not carry selected dispatch fallback mirrors "
        "for a selected typed RVV body route without dispatch fallback facts");
  }
  return llvm::Error::success();
}

llvm::Expected<plugin::VariantEmissionRole>
parseCandidateEmissionRole(const TargetArtifactCandidate &candidate) {
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DirectVariant))
    return plugin::VariantEmissionRole::DirectVariant;
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DispatchCase))
    return plugin::VariantEmissionRole::DispatchCase;
  if (candidate.role ==
      plugin::stringifyVariantEmissionRole(
          plugin::VariantEmissionRole::DispatchFallback))
    return plugin::VariantEmissionRole::DispatchFallback;

  return makeRVVTargetRouteError(
      llvm::Twine("candidate selected path role '") + candidate.role +
      "' is not supported by the RVV materialized EmitC artifact bridge");
}

llvm::Expected<tcrv::exec::VariantOp>
resolveCandidateSelectedVariant(const TargetArtifactCandidate &candidate) {
  if (!candidate.kernel)
    return makeRVVTargetRouteError(
        "candidate selected variant cannot be cross-checked without an "
        "enclosing tcrv.exec.kernel");
  tcrv::exec::KernelOp kernel = candidate.kernel;
  if (kernel.getBody().empty())
    return makeRVVTargetRouteError(
        "candidate selected variant cannot be cross-checked because the "
        "enclosing tcrv.exec.kernel body is empty");
  if (candidate.selectedVariant.empty())
    return makeRVVTargetRouteError(
        "candidate selected variant must be non-empty before RVV artifact "
        "route cross-check");

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
    if (variant && variant.getSymName() == candidate.selectedVariant)
      return variant;
  }

  return makeRVVTargetRouteError(
      llvm::Twine("candidate selected variant @") +
      candidate.selectedVariant +
      " must resolve to a direct sibling tcrv.exec.variant before RVV "
      "artifact route cross-check");
}

llvm::Error validateRVVRouteSourceProvenance(
    const conversion::emitc::TCRVEmitCLowerableRoute &route) {
  if (route.getSourceOpProvenance().size() != 1)
    return makeRVVTargetRouteError(
        "materialized EmitC route must carry exactly one RVV route "
        "source-op provenance entry");

  const conversion::emitc::TCRVEmitCSourceOpProvenance &source =
      route.getSourceOpProvenance().front();
  if (source.opName != plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName() ||
      source.role != "scope" ||
      source.opInterface != plugin::rvv::getRVVEmitCLowerableOpInterfaceName())
    return makeRVVTargetRouteError(
        "materialized EmitC route source-op provenance must identify "
        "tcrv_rvv.with_vl as the selected scope boundary through "
        "TCRVEmitCLowerableOpInterface");

  return llvm::Error::success();
}

llvm::Error validateRVVRouteABIMappings(
    const TargetArtifactCandidate &candidate,
    const conversion::emitc::TCRVEmitCLowerableRoute &route) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> routeParameters;
  for (const conversion::emitc::TCRVEmitCABIValueMapping &mapping :
       route.getABIMappings()) {
    if (mapping.valueName != mapping.parameter.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("materialized EmitC route ABI mapping for '") +
          mapping.parameter.cName +
          "' must use the same selected callable value name");
    routeParameters.push_back(mapping.parameter);
  }

  if (!support::runtimeABIParametersEqual(routeParameters,
                                          candidate.runtimeABIParameters))
    return makeRVVTargetRouteError(
        "materialized EmitC route ABI mappings must match the selected "
        "candidate runtime ABI parameters");

  return llvm::Error::success();
}

llvm::Expected<RVVSelectedVariantRouteValidation>
validateRVVSelectedVariantRouteAgreesWithCandidate(
    const TargetArtifactCandidate &candidate) {
  if (!candidate.kernel)
    return makeRVVTargetRouteError(
        "selected RVV materialized EmitC candidate requires an enclosing "
        "tcrv.exec.kernel so target export can validate the selected typed "
        "tcrv_rvv body before consuming route metadata");

  llvm::Expected<tcrv::exec::VariantOp> selectedVariant =
      resolveCandidateSelectedVariant(candidate);
  if (!selectedVariant)
    return selectedVariant.takeError();

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(candidate.kernel);
  if (!capabilities)
    return capabilities.takeError();

  llvm::Expected<plugin::VariantEmissionRole> role =
      parseCandidateEmissionRole(candidate);
  if (!role)
    return role.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route;
  plugin::VariantEmitCLowerableRequest request(
      *selectedVariant, candidate.kernel, *capabilities, *role);
  llvm::Expected<plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
      description =
          plugin::rvv::describeRVVSelectedBodyEmitCRoute(request, &route);
  if (!description) {
    llvm::Error error = description.takeError();
    std::string message = llvm::toString(std::move(error));
    return makeRVVTargetRouteError(
        llvm::Twine("selected typed RVV body could not build the "
                    "materialized EmitC route before candidate metadata "
                    "validation: ") +
        message);
  }
  if (llvm::Error error = route.verify()) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVTargetRouteError(
        llvm::Twine("rebuilt materialized EmitC route failed verification: ") +
        message);
  }

  if (llvm::Error error = validateRVVSegment2RoutePayloadFacts(
          route, *description))
    return std::move(error);
  if (llvm::Error error =
          validateRVVMAccRoutePayloadFacts(route, *description))
    return std::move(error);
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRoutePayloadFacts(
              route, *description))
    return std::move(error);
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRoutePayloadFacts(route,
                                                            *description))
    return std::move(error);
  if (llvm::Error error =
          validateRVVCompareSelectMaskRoutePayloadFacts(route, *description))
    return std::move(error);
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRoutePayloadFacts(route,
                                                            *description))
    return std::move(error);
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(route,
                                                              *description))
    return std::move(error);

  if (llvm::Error error = validateRVVRouteMetadataMirrorsSelectedBody(
          candidate, route, *description))
    return std::move(error);

  if (llvm::Error error = validateRVVRouteSourceProvenance(route))
    return std::move(error);
  if (llvm::Error error = validateRVVRouteABIMappings(candidate, route))
    return std::move(error);

  RVVSelectedVariantRouteValidation validation;
  validation.route = std::move(route);
  validation.description = std::move(*description);
  return validation;
}

llvm::Error rejectForbiddenRVVArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    llvm::StringRef key(entry.key);
    std::string lowerKeyStorage = key.lower();
    llvm::StringRef lowerKey(lowerKeyStorage);
    std::string lowerValueStorage = llvm::StringRef(entry.value).lower();
    llvm::StringRef lowerValue(lowerValueStorage);
    if (lowerKey.contains("element_count") ||
        lowerKey.contains("element-count") ||
        lowerKey.contains("descriptor") ||
        lowerKey.contains("direct_c") || lowerKey.contains("direct-c") ||
        lowerKey.contains("source_export") ||
        lowerKey.contains("source-export") ||
        lowerKey.contains("compute_body") ||
        lowerKey.contains("compute-body") ||
        lowerValue.contains("descriptor") ||
        lowerValue.contains("direct_c") ||
        lowerValue.contains("direct-c") ||
        lowerValue.contains("source_export") ||
        lowerValue.contains("source-export") ||
        lowerValue.contains("compute_body") ||
        lowerValue.contains("compute-body"))
      return makeRVVTargetRouteError(
          llvm::Twine("candidate artifact metadata key '") + key +
          "' attempts to reintroduce descriptor-driven computation, "
          "direct C/source-export authority, compute-body metadata, or "
          "hardcoded element-count residue");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVConstructionArtifactMetadata(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> rvvMetadata;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (llvm::StringRef(entry.key).starts_with("rvv_"))
      rvvMetadata.push_back(entry);
  }
  plugin::rvv::RVVSelectedBodyConstructionMetadataFacts facts =
      plugin::rvv::getRVVSelectedBodyConstructionMetadataFacts(description);
  return plugin::rvv::
      verifyRVVSelectedBodyConstructionArtifactMetadata(
          rvvMetadata, facts, "selected RVV materialized EmitC candidate");
}

llvm::Error validateRVVConfigArtifactMetadataMirrorsSelectedBody(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> actual;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (llvm::StringRef(entry.key).starts_with("tcrv_rvv."))
      actual.push_back(entry);
  }

  llvm::SmallVector<support::ArtifactMetadataEntry, 16> expected =
      plugin::rvv::getRVVSelectedBodyConfigArtifactMetadata(description);
  if (support::artifactMetadataEntriesEqual(actual, expected))
    return llvm::Error::success();

  if (actual.size() != expected.size())
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must carry exactly ") +
        llvm::Twine(expected.size()) +
        " tcrv_rvv selected-body config/runtime-VL artifact metadata entries "
        "derived from the provider route description");

  for (auto [index, pair] : llvm::enumerate(llvm::zip(actual, expected))) {
    const support::ArtifactMetadataEntry &got = std::get<0>(pair);
    const support::ArtifactMetadataEntry &want = std::get<1>(pair);
    if (got.key != want.key)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate tcrv_rvv selected-body metadata[") +
          llvm::Twine(index) + "] key must mirror provider route description "
                               "key '" +
          want.key + "' but was '" + got.key + "'");
    if (got.value != want.value)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate tcrv_rvv selected-body metadata key '") +
          want.key + "' must mirror provider route description value '" +
          want.value + "' but was '" + got.value + "'");
  }

  return makeRVVTargetRouteError(
      "candidate tcrv_rvv selected-body config/runtime-VL artifact metadata "
      "must mirror the provider route description");
}

llvm::Error validateRVVRuntimeAVLVLArtifactMetadata(
    const TargetArtifactCandidate &candidate,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (llvm::Error error = rejectForbiddenRVVArtifactMetadata(candidate))
    return error;
  if (llvm::Error error =
          validateRVVConstructionArtifactMetadata(candidate, description))
    return error;

  return validateRVVConfigArtifactMetadataMirrorsSelectedBody(candidate,
                                                             description);
}

llvm::Error validateRVVSelectedBodyTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  if (candidate.role == "dispatch fallback")
    return makeRVVTargetRouteError(
        "selected RVV materialized EmitC candidate must not be fallback-only");
  if (llvm::Error error = requireCandidateField(
          "origin", candidate.origin,
          plugin::rvv::getRVVConstructionManifest().family.pluginName))
    return error;
  if (llvm::Error error =
          requireCandidateField("emission kind", candidate.emissionKind,
                                plugin::rvv::getRVVSelectedBodyEmissionKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "lowering boundary", candidate.loweringBoundary,
          plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI kind", candidate.runtimeABIKind,
          plugin::rvv::getRVVSelectedBodyRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime glue role", candidate.runtimeGlueRole,
          plugin::rvv::getRVVSelectedBodyRuntimeGlueRole()))
    return error;

  llvm::Expected<RVVSelectedVariantRouteValidation> selectedRoute =
      validateRVVSelectedVariantRouteAgreesWithCandidate(candidate);
  if (!selectedRoute)
    return selectedRoute.takeError();

  if (!support::runtimeABIParametersEqual(
          candidate.runtimeABIParameters,
          selectedRoute->description.runtimeABIParameters))
    return makeRVVTargetRouteError(
        "candidate runtime ABI parameters must mirror the provider-derived "
        "selected-body runtime ABI signature");

  if (llvm::Error error = requireCandidateField(
          "route id", candidate.routeID,
          selectedRoute->description.targetArtifactRouteID))
    return error;
  if (llvm::Error error = requireCandidateField(
          "artifact kind", candidate.artifactKind,
          selectedRoute->description.targetArtifactKind))
    return error;

  llvm::StringRef selectedBodyRuntimeABIName =
      selectedRoute->description.runtimeABIName;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI", candidate.runtimeABI, selectedBodyRuntimeABIName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI name", candidate.runtimeABIName,
          selectedBodyRuntimeABIName))
    return error;
  if (llvm::Error error = validateRVVRuntimeAVLVLArtifactMetadata(
          candidate, selectedRoute->description))
    return error;
  return llvm::Error::success();
}

void appendRVVConfigVLMetadataEvidence(
    llvm::SmallVectorImpl<MaterializedEmitCHeaderArtifactMetadataEvidence>
        &out) {
  constexpr llvm::StringLiteral kRVVMetadataPrefix("tcrv_rvv.");
  for (const support::ArtifactMetadataEntry &entry :
       tcrv::rvv::getRVVSelectedBodyConfigArtifactMetadata()) {
    llvm::StringRef key(entry.key);
    llvm::StringRef commentName = key;
    if (key.starts_with(kRVVMetadataPrefix))
      commentName = key.drop_front(kRVVMetadataPrefix.size());
    out.push_back({commentName, key, /*expectedValue=*/"",
                   /*allowDynamicValue=*/true});
  }
}

llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence, 32>
buildRVVSelectedBodyHeaderMetadataEvidence() {
  llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence, 32>
      evidence;
  evidence.append({
      {"source_ops", plugin::rvv::getRVVSourceOpsMetadataName(),
       plugin::rvv::getRVVSelectedBodySourceOps()},
      {"source_roles", plugin::rvv::getRVVSourceRolesMetadataName(),
       plugin::rvv::getRVVSelectedBodySourceRoles()},
      {"source_op_interface",
       plugin::rvv::getRVVSourceOpInterfaceMetadataName(),
       plugin::rvv::getRVVEmitCLowerableOpInterfaceName()},
      {"construction_protocol",
       plugin::rvv::getRVVConstructionProtocolMetadataName(),
       plugin::rvv::getRVVConstructionManifest().protocolVersion},
      {"extension_archetype",
       plugin::rvv::getRVVConstructionArchetypeMetadataName(),
       plugin::rvv::getRVVConstructionManifest().archetype},
      {"semantic_role_graph", plugin::rvv::getRVVSemanticRoleGraphMetadataName(),
       plugin::rvv::getRVVConstructionManifest().semanticRoleGraph},
      {"common_interface_realization",
       plugin::rvv::getRVVCommonInterfaceRealizationMetadataName(),
       plugin::rvv::getRVVConstructionArtifactInterfaceRealization()},
      {"typed_role_realization",
       plugin::rvv::getRVVTypedRoleRealizationMetadataName(),
       plugin::rvv::getRVVArtifactTypedRoleRealizationSummary()},
      {"emitc_route_mapping",
       plugin::rvv::getRVVEmitCRouteMappingMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID()},
      {"target_artifact_route",
       plugin::rvv::getRVVTargetArtifactRouteMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID()},
      {"target_artifact_kind",
       plugin::rvv::getRVVTargetArtifactKindMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactKind()},
      {"evidence_profile", plugin::rvv::getRVVEvidenceProfileMetadataName(),
       plugin::rvv::getRVVConstructionManifest().evidenceProfile},
      {"bundle_component_group",
       plugin::rvv::getRVVBundleComponentGroupMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactMapping()
           .bundleComponentGroup},
      {"object_handoff", plugin::rvv::getRVVObjectHandoffMetadataName(),
       plugin::rvv::getRVVSelectedBodyTargetArtifactMapping()
           .objectHandoffKind},
  });
  appendRVVConfigVLMetadataEvidence(evidence);
  evidence.append({
      {"compare_predicate_kind", "tcrv_rvv.compare_predicate_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"memory_form", "tcrv_rvv.memory_form", "",
       /*allowDynamicValue=*/true},
      {"strided_memory_layout", "tcrv_rvv.strided_memory_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"lhs_stride_source", "tcrv_rvv.lhs_stride_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"rhs_stride_source", "tcrv_rvv.rhs_stride_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_memory_form", "tcrv_rvv.source_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"destination_memory_form", "tcrv_rvv.destination_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_role", "tcrv_rvv.mask_role", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_source", "tcrv_rvv.mask_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_memory_form", "tcrv_rvv.mask_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_route_family_plan",
       "tcrv_rvv.mask_tail_policy_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_owner", "tcrv_rvv.mask_tail_policy_owner", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"secondary_compare_predicate_kind",
       "tcrv_rvv.secondary_compare_predicate_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_composition", "tcrv_rvv.mask_composition", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"inactive_lane_contract", "tcrv_rvv.inactive_lane_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"masked_passthrough_layout", "tcrv_rvv.masked_passthrough_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"masked_memory_layout", "tcrv_rvv.masked_memory_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"select_layout", "tcrv_rvv.select_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"indexed_memory_layout", "tcrv_rvv.indexed_memory_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"index_source", "tcrv_rvv.index_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"index_eew", "tcrv_rvv.index_eew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"offset_unit", "tcrv_rvv.offset_unit", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"index_uniqueness", "tcrv_rvv.index_uniqueness", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"indexed_data_memory_form", "tcrv_rvv.indexed_data_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"indexed_destination_memory_form",
       "tcrv_rvv.indexed_destination_memory_form", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_sew", "tcrv_rvv.source_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"source_lmul", "tcrv_rvv.source_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dest_sew", "tcrv_rvv.dest_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"dest_lmul", "tcrv_rvv.dest_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"conversion_relation", "tcrv_rvv.conversion_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulator_sew", "tcrv_rvv.accumulator_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulator_lmul", "tcrv_rvv.accumulator_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"result_sew", "tcrv_rvv.result_sew", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"result_lmul", "tcrv_rvv.result_lmul", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"macc_accumulator_layout", "tcrv_rvv.macc_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"macc_result_layout", "tcrv_rvv.macc_result_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_macc_accumulator_layout",
       "tcrv_rvv.widening_macc_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_macc_result_layout",
       "tcrv_rvv.widening_macc_result_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_macc_relation", "tcrv_rvv.widening_macc_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_accumulator_layout",
       "tcrv_rvv.widening_dot_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_result_layout", "tcrv_rvv.widening_dot_result_layout",
       "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_relation", "tcrv_rvv.widening_dot_relation", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_dot_reduction_store_vl",
       "tcrv_rvv.widening_dot_reduction_store_vl", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"reduction_accumulator_layout",
       "tcrv_rvv.reduction_accumulator_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"reduction_result_layout", "tcrv_rvv.reduction_result_layout", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"reduction_store_vl", "tcrv_rvv.reduction_store_vl", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"target_leaf_profile", "tcrv_rvv.target_leaf_profile", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"runtime_control_plan", "tcrv_rvv.runtime_control_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"provider_supported_mirror", "tcrv_rvv.provider_supported_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"target_capability_provider_mirror",
       "tcrv_rvv.target_capability_provider_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"target_capability_legality_mirror",
       "tcrv_rvv.target_capability_legality_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"selected_dispatch_case_mirror",
       "tcrv_rvv.selected_dispatch_case_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"selected_dispatch_fallback_mirror",
       "tcrv_rvv.selected_dispatch_fallback_mirror", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"route_operand_binding_plan",
       "tcrv_rvv.route_operand_binding_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"route_operand_binding_operands",
       "tcrv_rvv.route_operand_binding_operands", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"exec_abi_bindings", "tcrv_rvv.exec_abi_bindings", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"contraction_route_family_plan",
       "tcrv_rvv.contraction_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_select_route_family_plan",
       "tcrv_rvv.computed_mask_select_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_select_mask_producer_source",
       "tcrv_rvv.computed_mask_select_mask_producer_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_memory_route_family_plan",
       "tcrv_rvv.computed_mask_memory_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"computed_mask_memory_mask_producer_source",
       "tcrv_rvv.computed_mask_memory_mask_producer_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_route_family_plan",
       "tcrv_rvv.mask_tail_policy_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"mask_tail_policy_owner", "tcrv_rvv.mask_tail_policy_owner", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"base_memory_movement_route_family_plan",
       "tcrv_rvv.base_memory_movement_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"elementwise_arithmetic_route_family_plan",
       "tcrv_rvv.elementwise_arithmetic_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"scalar_broadcast_elementwise_route_family_plan",
       "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"scalar_broadcast_macc_route_family_plan",
       "tcrv_rvv.scalar_broadcast_macc_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"plain_macc_route_family_plan",
       "tcrv_rvv.plain_macc_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"plain_compare_select_route_family_plan",
       "tcrv_rvv.plain_compare_select_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"segment2_memory_route_family_plan",
       "tcrv_rvv.segment2_memory_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"widening_conversion_route_family_plan",
       "tcrv_rvv.widening_conversion_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_route_family_plan",
       "tcrv_rvv.standalone_reduction_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_source_vector_type",
       "tcrv_rvv.standalone_reduction_source_vector_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_source_vector_c_type",
       "tcrv_rvv.standalone_reduction_source_vector_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_scalar_result_vector_type",
       "tcrv_rvv.standalone_reduction_scalar_result_vector_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_scalar_result_vector_c_type",
       "tcrv_rvv.standalone_reduction_scalar_result_vector_c_type", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"standalone_reduction_scalar_result_runtime_boundary",
       "tcrv_rvv.standalone_reduction_scalar_result_runtime_boundary", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_route_family_plan",
       "tcrv_rvv.accumulation_route_family_plan", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_compute_suffix",
       "tcrv_rvv.accumulation_compute_suffix", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_mask_producer_source",
       "tcrv_rvv.accumulation_mask_producer_source", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_accumulator_contract",
       "tcrv_rvv.accumulation_accumulator_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_result_contract",
       "tcrv_rvv.accumulation_result_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"accumulation_scalar_carry_contract",
       "tcrv_rvv.accumulation_scalar_carry_contract", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"required_header_declarations",
       "tcrv_rvv.required_header_declarations", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"c_type_mapping", "tcrv_rvv.c_type_mapping", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"segment2_update_arithmetic_kind",
       "tcrv_rvv.segment2_update_arithmetic_kind", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
      {"inactive_lane_zeroing_requirement",
       "tcrv_rvv.inactive_lane_zeroing_requirement", "",
       /*allowDynamicValue=*/true, /*optional=*/true},
  });
  return evidence;
}

ConstructionTemplateArtifactAdapterConfig
getRVVSelectedBodyArtifactAdapterConfig();

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportConstructionTemplateEmitCToCpp(
      module, os, getRVVSelectedBodyArtifactAdapterConfig());
}

llvm::Error compileRVVGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clang = llvm::sys::findProgramByName("clang");
  if (!clang)
    clang = llvm::sys::findProgramByName(
        "clang", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clang)
    return makeRVVTargetRouteError(
        llvm::Twine("requires clang on PATH or a standard LLVM tools path for "
                    "RISC-V object packaging: ") +
        clang.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-materialized-emitc", "cpp", sourceFD, sourcePath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeRVVTargetRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-materialized-emitc-clang", "stderr", stderrFD,
          stderrPath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary clang stderr file: ") +
        error.message());
  {
    llvm::raw_fd_ostream stderrOS(stderrFD, /*shouldClose=*/true);
    stderrOS.close();
  }

  llvm::SmallVector<llvm::StringRef, 12> args = {
      *clang,
      "-target",
      "riscv64",
      "-O2",
      "-march=rv64gcv",
      "-mabi=lp64d",
      "-c",
      sourcePath.path,
      "-o",
      objectPath.path};
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), llvm::StringRef(), llvm::StringRef(stderrPath.path)};
  std::string executeError;
  bool executionFailed = false;
  int result = llvm::sys::ExecuteAndWait(
      *clang, args, std::nullopt, redirects, /*SecondsToWait=*/30,
      /*MemoryLimit=*/0, &executeError, &executionFailed);
  if (executionFailed || result != 0) {
    std::string stderrText;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderrBuffer =
        llvm::MemoryBuffer::getFile(stderrPath.path);
    if (stderrBuffer)
      stderrText = (*stderrBuffer)->getBuffer().take_front(512).str();
    return makeRVVTargetRouteError(
        llvm::Twine("clang failed to package materialized EmitC C/C++ source "
                    "as a RISC-V RVV relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeRVVTargetRouteError(
        llvm::Twine("failed to read generated RISC-V object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeRVVTargetRouteError("generated RISC-V object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig getRVVSelectedBodyArtifactConfig() {
  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID();
  config.artifactKind = plugin::rvv::getRVVSelectedBodyTargetArtifactKind();
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription = "RVV selected-body materialized EmitC target "
                            "artifact bridge for the bounded i32 arithmetic "
                            "selected-body slice";
  config.candidateValidationFn =
      validateRVVSelectedBodyTargetArtifactCandidate;
  config.routeBuilderFn =
      plugin::rvv::buildRVVSelectedBodyEmitCLowerableRoute;
  return config;
}

ConstructionTemplateArtifactAdapterConfig
getRVVSelectedBodyArtifactAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stddef.h", "stdint.h"};
  static const ConstructionTemplateSelectedBoundaryAttributeExpectation
      kBoundaryAttributeExpectations[] = {
          {plugin::rvv::getRVVConstructionProtocolMetadataName(),
           plugin::rvv::getRVVConstructionProtocolVersion(), {}},
      };
  static const llvm::SmallVector<MaterializedEmitCHeaderArtifactMetadataEvidence,
                                 32>
      kMetadataEvidence = buildRVVSelectedBodyHeaderMetadataEvidence();
  static const llvm::SmallVector<support::RuntimeABIParameter, 4>
      kRuntimeABIParameters =
          plugin::rvv::getRVVSelectedBodyConstructionRuntimeABIParameters();

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  const plugin::rvv::RVVSelectedBodyTargetArtifactMapping &mapping =
      getRVVTargetMapping();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute = getRVVSelectedBodyArtifactConfig();
  config.selectedRoute.routeDescription =
      "RVV selected-body construction-template materialized EmitC artifact "
      "adapter for the bounded i32 arithmetic selected-body slice";
  config.headerRouteID = mapping.headerRouteID;
  config.headerArtifactKind = mapping.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
  config.headerGuard = "TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H";
  config.evidencePrefix = "tianchenrv.rvv";
  config.includes = kHeaderIncludes;
  config.selectedVariant = "";
  config.emissionKind = plugin::rvv::getRVVSelectedBodyEmissionKind();
  config.loweringBoundary =
      plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName();
  config.runtimeABIKind =
      plugin::rvv::getRVVSelectedBodyRuntimeABIKind();
  config.runtimeGlueRole =
      plugin::rvv::getRVVSelectedBodyRuntimeGlueRole();
  config.allowDynamicRuntimeABIIdentity = true;
  config.runtimeABIParameters = kRuntimeABIParameters;
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = mapping.bundleComponentGroup;
  config.externalABIName = "";
  config.handoffKind = mapping.objectHandoffKind;
  config.selectedObjectDescription = "RVV materialized EmitC candidate";
  config.selectedLoweringBoundary.required = true;
  config.selectedLoweringBoundary.boundaryDescription =
      "selected RVV construction-template artifact boundary";
  config.selectedLoweringBoundary.status =
      plugin::rvv::getRVVLoweringBoundaryStatus();
  config.selectedLoweringBoundary.sourceKernelAttrName =
      plugin::rvv::getRVVSourceKernelAttrName();
  config.selectedLoweringBoundary.selectedVariantAttrName =
      plugin::rvv::getRVVSelectedVariantAttrName();
  config.selectedLoweringBoundary.originAttrName =
      plugin::rvv::getRVVOriginAttrName();
  config.selectedLoweringBoundary.roleAttrName =
      plugin::rvv::getRVVSelectedPathRoleAttrName();
  config.selectedLoweringBoundary.statusAttrName =
      plugin::rvv::getRVVStatusAttrName();
  config.selectedLoweringBoundary.requiredCapabilitiesAttrName =
      plugin::rvv::getRVVRequiredCapabilitiesAttrName();
  config.selectedLoweringBoundary.extraStringAttributes =
      kBoundaryAttributeExpectations;
  config.selectedLoweringBoundary.searchSelectedVariantBody = true;
  config.objectPackagerFn = compileRVVGeneratedSourceToObject;
  return config;
}

llvm::Error exportRVVSelectedBodyTargetArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  return exportConstructionTemplateObjectArtifact(
      module, os, getRVVSelectedBodyArtifactAdapterConfig());
}

llvm::Error exportRVVSelectedBodyHeaderArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  return exportConstructionTemplateHeaderArtifact(
      module, os, getRVVSelectedBodyArtifactAdapterConfig());
}

llvm::Error registerRVVSelectedBodyTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  return registerConstructionTemplateArtifactAdapterExporters(
      registry, getRVVSelectedBodyArtifactAdapterConfig(),
      exportRVVSelectedBodyTargetArtifact,
      exportRVVSelectedBodyHeaderArtifact);
}

} // namespace

llvm::StringRef getRVVMaterializedEmitCTargetArtifactRouteID() {
  return plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID();
}

llvm::StringRef getRVVMaterializedEmitCHeaderArtifactRouteID() {
  return getRVVTargetMapping().headerRouteID;
}

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getRVVManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerRVVSelectedBodyTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerRVVSelectedBodyTargetArtifactExporter));
}

llvm::Error
configureRVVTargetSupportExtensionBundle(plugin::ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(
      plugin::rvv::getRVVSelectedBodyLoweringBoundaryOpName());
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  llvm::StringRef routeID = getRVVTargetMapping().emitCToCppTranslateRouteID;
  if (registry.lookup(routeID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      routeID,
      "export a materialized RVV EmitC module through the MLIR EmitC "
      "C/C++ emitter",
      exportMaterializedRVVEmitCToCpp));
}

} // namespace tianchenrv::target::rvv
